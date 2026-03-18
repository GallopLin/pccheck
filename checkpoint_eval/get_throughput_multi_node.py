import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/transformers/examples/pytorch/language-modeling"
this_dir = f"{home_dir}/pccheck/artifact_evaluation/evaluation/throughput"
cfreqs = [10]
iters = 50
WARMUP = 3

label_dict = {
    "cfreq": "CheckFreq",
    "gpm": "GPM",
    "pccheck": "PCcheck",
    "gemini": "Gemini",
    "multistream": "MultiStream",
}

ms_checkpoint_dir = f"{home_dir}/ms_checkpoints_bench"

def create_files(ip1):

    with open(f"{script_dir}/hostfile", "w") as f:
        f.write(f"{ip1} slots=4\n")
    f.close()

    with open(f"{home_dir}/.deepspeed_env", "w") as f:  # TODO
        f.write(f"GEMINI_MASTER_ADDR={ip1}\n")
        f.write(f"GEMINI_MASTER_PORT=1235\n")
        f.write(f"PCCHECK_COORDINATOR={ip1}\n")
    f.close()


def run(ip1):
    os.makedirs(f"{this_dir}/opt_27", exist_ok=True)

    def run_and_log(cmd, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"[CMD] {cmd}\n\n")
            f.flush()
            ret = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            f.write(f"\n[EXIT_CODE] {ret.returncode}\n")
        return ret.returncode

    # # run cfreq
    # print("Run for CheckFreq")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && {sys.executable} -m deepspeed.launcher.runner --num_gpus=4 run_clm_pp_cfreq.py --deepspeed ds_config.json --ds_config ds_config.json --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters}"
    #     run_and_log(proc, f"{this_dir}/opt_27/log_opt27_cfreq_{cf}.txt")

    # # run gpm
    # print("Run for GPM")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && {sys.executable} -m deepspeed.launcher.runner --num_gpus=4 run_clm_pp_gpm.py --deepspeed ds_config.json --ds_config ds_config.json --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} > {this_dir}/opt_27/log_opt27_gpm_{cf}.txt"
    #     os.system(proc)

    # # run gemini
    # print("Run for Gemini")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && {sys.executable} -m deepspeed.launcher.runner --num_gpus=4 run_clm_pp_gemini.py --deepspeed ds_config.json --ds_config ds_config.json --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} > {this_dir}/opt_27/log_opt27_gemini_{cf}.txt"
    #     os.system(proc)


    # # run pccheck
    print("Run for PCCheck")
    for cf in cfreqs:
    print(f"Checkpoint Frequency {cf}")
    proc = f"export PYTHONPATH=/root/pccheck && cd {script_dir} && {sys.executable} -m deepspeed.launcher.runner --num_gpus=4 run_clm_pp_pccheck.py --deepspeed /root/pccheck/checkpoint_eval/models/llm_distr/ds_config.json --ds_config /root/pccheck/checkpoint_eval/models/llm_distr/ds_config.json --model_name_or_path facebook/opt-2.7b --output_dir ./output_opt27_4gpu --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path} --max_async 2 --num_threads 2"
    run_and_log(proc, f"{this_dir}/opt_27/log_opt27_pccheck_{cf}.txt")

    # # run multistream
    print("Run for MultiStream")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && {sys.executable} -m deepspeed.launcher.runner --num_gpus=4 run_clm_pp_multistream.py --deepspeed ds_config.json --ds_config ds_config.json --model_name_or_path facebook/opt-350m --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path_stream} --max_async 2 --num_threads 2 --checkpoint_dir {ms_checkpoint_dir}"
        run_and_log(proc, f"{this_dir}/opt_27/log_opt27_multistream_{cf}.txt")

    # # run multistream
    print("Run for MultiStream")


def collect():
    def get_exec_throughput(input_file, baseline):
        exec_time  = 0.0
        extra_time = 0.0
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    exec_time = float(tokens[-2])
                elif 'MMAP/UMAP' in line:
                    tokens = line.split(" ")
                    extra_time = float(tokens[-2])/1000 # convert in sec
        exec_time -= extra_time
        if exec_time <= 0:
            raise RuntimeError(
                f"Invalid EXECUTION TIME parsed from {input_file}: {exec_time}. "
                "Please make sure the run finished successfully and log contains 'EXECUTION TIME'."
            )
        thr = (iters-WARMUP)/exec_time
        return thr

    throughput_dict = {}
    throughput_list = []

    for baseline in ["cfreq", "gpm", "gemini", "pccheck", "multistream"]:
        baseline_thr = []
        for cf in cfreqs:
            input_file = f"{this_dir}/opt_27/log_opt27_{baseline}_{cf}.txt"
            thr = get_exec_throughput(input_file, baseline)
            baseline_thr.append(thr)
        throughput_list.append(baseline_thr)
        throughput_dict[label_dict[baseline]] = baseline_thr

    print(throughput_list)
    column_header = [str(x) for x in cfreqs]
    index_header = ["CheckFreq", "GPM", "Gemini", "PCcheck", "MultiStream"]
    df = pd.DataFrame(throughput_list, columns = column_header, index = index_header)
    df.to_csv(f'fig8_opt27.csv')
    return throughput_dict


def plot(data):
    colors = ['#18384F', '#4392B8', '#E27733', '#A7B972', '#C75B7A']
    label_font_size = 36
    num_methods = len(data)
    width = 0.8 / max(num_methods, 1)
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(cfreqs))
    bars = []

    for method_id, (method_key, method_data) in enumerate(data.items()):
        bar = ax.bar(
            x + width * method_id, method_data, width,
            label=method_key,
            align='edge',
            color=colors[method_id % len(colors)]
        )
        bars.append(bar)

    plt.yticks(fontsize=label_font_size)

    # 仅在存在多个 checkpoint interval 时，绘制 cfreq=10 的参考线
    if len(cfreqs) > 1 and "MultiStream" in data:
        ax.plot(x + 2 * width, [data["MultiStream"][0]] * len(x), color='#C75B7A', marker="^", linewidth=3, markersize=8, linestyle='--', label="MultiStream (cfreq=10)")

    x_tick_positions = x + width * num_methods / 2
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=cfreqs, fontsize=label_font_size,
    )
    plt.yticks(fontsize=label_font_size)

    ax.set_ylabel('Throughput (iterations/sec)', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint interval(iterations)', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles, labels, loc='upper left', ncol=3, fontsize=label_font_size - 6, bbox_to_anchor=(-0.025, 1.25))
    plt.savefig(f"fig8_opt27.png", bbox_inches="tight", dpi=500, pad_inches=0.1)


if __name__ == "__main__":
    # ip1 = "127.0.0.1"
    # create_files(ip1)
    # run(ip1)
    data = collect()
    plot(data)
