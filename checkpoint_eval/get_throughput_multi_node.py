import os
import sys
import json
import subprocess
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/download/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
lib_path_stream = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/code/transformers/examples/pytorch/language-modeling"
this_dir = f"{home_dir}/code/pccheck/artifact_evaluation/evaluation/throughput"
cfreqs = [1] # 0, 1, 10, 25, 50, 100
iters = 210 #210
WARMUP = 3

label_dict = {
    "cfreq": "CheckFreq",
    "gpm": "GPM",
    "pccheck": "PCcheck",
    "gemini": "Gemini",
    "multistream": "MultiStream",
}

ms_checkpoint_dir = f"{home_dir}/ms_checkpoints_bench"


def _parse_gpu_list(gpu_spec):
    return [x.strip() for x in gpu_spec.split(",") if x.strip()]


# OPT-2.7B 仍保持原有 2-stage 配置，避免和未改造的 entrypoint 不一致。
OPT_ACTIVE_GPUS = os.environ.get("PCCHECK_OPT_ACTIVE_GPUS", "1,7")
OPT_NUM_GPUS = len(_parse_gpu_list(OPT_ACTIVE_GPUS))
OPT_PP_DEGREE = int(os.environ.get("PCCHECK_OPT_PP_DEGREE", "2"))

# Bloom-7B 默认切到 4 卡 + 4-stage pipeline，降低单卡显存峰值。
BLOOM_ACTIVE_GPUS = os.environ.get("PCCHECK_BLOOM_ACTIVE_GPUS", "1,4,6,7")
BLOOM_NUM_GPUS = len(_parse_gpu_list(BLOOM_ACTIVE_GPUS))
BLOOM_PP_DEGREE = int(os.environ.get("PCCHECK_BLOOM_PP_DEGREE", str(BLOOM_NUM_GPUS)))

def create_files(ip1):

    with open(f"{script_dir}/hostfile", "w") as f:
        f.write(f"{ip1} slots={max(OPT_NUM_GPUS, BLOOM_NUM_GPUS)}\n")

    with open(f"{home_dir}/.deepspeed_env", "w") as f:  # TODO
        f.write(f"GEMINI_MASTER_ADDR={ip1}\n")
        f.write(f"GEMINI_MASTER_PORT=1235\n")
        f.write(f"PCCHECK_COORDINATOR={ip1}\n")


def run(ip1):
    os.makedirs(f"{this_dir}/opt_27", exist_ok=True)
    pccheck_root = f"{home_dir}/code/pccheck"

    if OPT_NUM_GPUS % OPT_PP_DEGREE != 0:
        raise ValueError(
            f"NUM_GPUS={OPT_NUM_GPUS} 与 PP_DEGREE={OPT_PP_DEGREE} 不整除，"
            "当前脚本的 pipeline 配置无法工作。"
        )

    def _prepare_runtime_ds_config():
        """根据当前 GPU 数生成运行时 ds_config，修正 train_batch_size 断言。"""
        base_cfg = f"{script_dir}/ds_config.json"
        out_cfg = f"{this_dir}/opt_27/ds_config_runtime_{OPT_NUM_GPUS}g_pp{OPT_PP_DEGREE}.json"

        with open(base_cfg, "r") as f:
            cfg = json.load(f)

        micro = int(cfg.get("train_micro_batch_size_per_gpu", 1))
        grad = int(cfg.get("gradient_accumulation_steps", 1))
        dp_degree = OPT_NUM_GPUS // OPT_PP_DEGREE

        # DeepSpeed 断言：train_batch_size == micro * grad * data_parallel_size
        cfg["train_batch_size"] = micro * grad * dp_degree

        with open(out_cfg, "w") as f:
            json.dump(cfg, f, indent=2)

        print(
            f"[INFO] Runtime ds_config: {out_cfg} | "
            f"train_batch_size={cfg['train_batch_size']} "
            f"(micro={micro}, grad={grad}, dp={dp_degree})"
        )
        return out_cfg

    runtime_ds_config = _prepare_runtime_ds_config()

    def _kill_stale_deepspeed():
        """清理残留分布式训练进程，防止 local_rank 子进程残留占用 GPU 显存。"""
        py_bin_pat = re.escape(sys.executable)
        stale_patterns = [
            r"deepspeed\.launcher",
            r"torch\.distributed",
            r"run_clm_pp_.*--local_rank",
            rf"{py_bin_pat} -c from multiprocessing\.spawn import spawn_main",
            rf"{py_bin_pat} -c from multiprocessing\.resource_tracker import main;main",
        ]
        for pattern in stale_patterns:
            subprocess.run(
                f"pkill -9 -f '{pattern}' 2>/dev/null",
                shell=True,
                executable="/bin/bash",
            )
        # 等待进程与端口释放
        time.sleep(6)

    def run_and_log(cmd, log_path):
        _kill_stale_deepspeed()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # 注入 PYTHONPATH，确保 checkpoint_eval 包可被 import
        env = os.environ.copy()
        env["PYTHONPATH"] = pccheck_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with open(log_path, "w") as f:
            f.write(f"[CMD] {cmd}\n\n")
            f.flush()
            ret = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            f.write(f"\n[EXIT_CODE] {ret.returncode}\n")
        # 即使 launcher 异常退出，也强制清理可能遗留的 local_rank 子进程
        _kill_stale_deepspeed()
        return ret.returncode

    # # run cfreq
    # print("Run for CheckFreq")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={OPT_ACTIVE_GPUS} {sys.executable} -m deepspeed.launcher.runner run_clm_pp_cfreq.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters}"
    #     run_and_log(proc, f"{this_dir}/opt_27/log_opt27_cfreq_{cf}.txt")

    # # run gpm
    print("Run for GPM")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        # 对齐其他 baseline：使用运行时 ds_config + 指定可见 GPU，避免固定 4 卡触发 GPM 大映射失败。
        proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={OPT_ACTIVE_GPUS} PCCHECK_GPM_MAX_MAPPED_BYTES=25769803776 {sys.executable} -m deepspeed.launcher.runner run_clm_pp_gpm.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters}"
        run_and_log(proc, f"{this_dir}/opt_27/log_opt27_gpm_{cf}.txt")

    # # run gemini
    # print("Run for Gemini")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && {sys.executable} -m deepspeed.launcher.runner --num_gpus=4 run_clm_pp_gemini.py --deepspeed ds_config.json --ds_config ds_config.json --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} > {this_dir}/opt_27/log_opt27_gemini_{cf}.txt"
    #     os.system(proc)


    # # run pccheck
    # print("Run for PCCheck")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={OPT_ACTIVE_GPUS} {sys.executable} -m deepspeed.launcher.runner run_clm_pp_pccheck.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path} --max_async 2 --num_threads 8 --checkpoint_dir {this_dir}/opt_27/pccheck_checkpoints"
    #     run_and_log(proc, f"{this_dir}/opt_27/log_opt27_pccheck_{cf}.txt")

    # run multistream
    print("Run for MultiStream")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={OPT_ACTIVE_GPUS} {sys.executable} -m deepspeed.launcher.runner run_clm_pp_multistream.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path_stream} --max_async 2 --num_threads 8 --checkpoint_dir {ms_checkpoint_dir}"
        run_and_log(proc, f"{this_dir}/opt_27/log_opt27_multistream_{cf}.txt")


def run_bloom7b(ip1):
    os.makedirs(f"{this_dir}/bloom_7b", exist_ok=True)
    pccheck_root = f"{home_dir}/code/pccheck"

    if BLOOM_NUM_GPUS % BLOOM_PP_DEGREE != 0:
        raise ValueError(
            f"NUM_GPUS={BLOOM_NUM_GPUS} 与 PP_DEGREE={BLOOM_PP_DEGREE} 不整除，"
            "当前脚本的 pipeline 配置无法工作。"
        )

    def _prepare_runtime_ds_config_bloom():
        """根据当前 GPU 数生成 Bloom 运行时 ds_config，修正 train_batch_size 断言。"""
        base_cfg = f"{script_dir}/ds_config.json"
        out_cfg = f"{this_dir}/bloom_7b/ds_config_runtime_{BLOOM_NUM_GPUS}g_pp{BLOOM_PP_DEGREE}.json"

        with open(base_cfg, "r") as f:
            cfg = json.load(f)

        micro = int(cfg.get("train_micro_batch_size_per_gpu", 1))
        grad = int(cfg.get("gradient_accumulation_steps", 1))
        dp_degree = BLOOM_NUM_GPUS // BLOOM_PP_DEGREE

        cfg["train_batch_size"] = micro * grad * dp_degree

        with open(out_cfg, "w") as f:
            json.dump(cfg, f, indent=2)

        print(
            f"[INFO] Bloom runtime ds_config: {out_cfg} | "
            f"train_batch_size={cfg['train_batch_size']} "
            f"(micro={micro}, grad={grad}, dp={dp_degree})"
        )
        return out_cfg

    runtime_ds_config = _prepare_runtime_ds_config_bloom()

    def _kill_stale_deepspeed():
        py_bin_pat = re.escape(sys.executable)
        stale_patterns = [
            r"deepspeed\.launcher",
            r"torch\.distributed",
            r"run_clm_pp_.*--local_rank",
            rf"{py_bin_pat} -c from multiprocessing\.spawn import spawn_main",
            rf"{py_bin_pat} -c from multiprocessing\.resource_tracker import main;main",
        ]
        for pattern in stale_patterns:
            subprocess.run(
                f"pkill -9 -f '{pattern}' 2>/dev/null",
                shell=True,
                executable="/bin/bash",
            )
        time.sleep(6)

    def run_and_log(cmd, log_path):
        _kill_stale_deepspeed()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = pccheck_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with open(log_path, "w") as f:
            f.write(f"[CMD] {cmd}\n\n")
            f.flush()
            ret = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            f.write(f"\n[EXIT_CODE] {ret.returncode}\n")
        # 即使 launcher 异常退出，也强制清理可能遗留的 local_rank 子进程
        _kill_stale_deepspeed()
        return ret.returncode

    bloom_model = "bigscience/bloom-7b1"

    # run pccheck for Bloom 7B
    # print("Run PCCheck for Bloom 7B")
    # for cf in cfreqs:
    #     print(f"Checkpoint Frequency {cf}")
    #     proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={BLOOM_ACTIVE_GPUS} {sys.executable} -m deepspeed.launcher.runner run_clm_pp_pccheck.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path {bloom_model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path} --max_async 2 --num_threads 8 --pp_degree {BLOOM_PP_DEGREE} --checkpoint_dir {this_dir}/bloom_7b/pccheck_checkpoints"
    #     run_and_log(proc, f"{this_dir}/bloom_7b/log_bloom7b_pccheck_{cf}.txt")

    # run checkfreq for Bloom 7B
    print("Run CheckFreq for Bloom 7B")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={BLOOM_ACTIVE_GPUS} {sys.executable} -m deepspeed.launcher.runner run_clm_pp_cfreq_bloom.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path {bloom_model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --pp_degree {BLOOM_PP_DEGREE}"
        run_and_log(proc, f"{this_dir}/bloom_7b/log_bloom7b_cfreq_{cf}.txt")

    # run multistream for Bloom 7B
    print("Run MultiStream for Bloom 7B")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && CUDA_VISIBLE_DEVICES={BLOOM_ACTIVE_GPUS} {sys.executable} -m deepspeed.launcher.runner run_clm_pp_multistream_bloom.py --deepspeed {runtime_ds_config} --ds_config {runtime_ds_config} --model_name_or_path {bloom_model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path_stream} --max_async 2 --num_threads 8 --checkpoint_dir {ms_checkpoint_dir} --pp_degree {BLOOM_PP_DEGREE}"
        run_and_log(proc, f"{this_dir}/bloom_7b/log_bloom7b_multistream_{cf}.txt")


def _expected_checkpoints(cfreq, baseline):
    if cfreq <= 0:
        return 0

    # 与 run_clm_pp_pccheck.py / run_clm_pp_multistream.py 的训练循环保持一致：
    # if (step == warmup) or (cfreq > 0 and steps_since_checkp == cfreq - 1)
    # 特别是 cfreq=1 时，会在每个 step 触发 checkpoint，总数应为 iters。
    if baseline in {"pccheck", "multistream"}:
        steps_since_checkp = 0
        checkpoints = 0
        for step in range(iters):
            if (step == WARMUP) or (steps_since_checkp == cfreq - 1):
                checkpoints += 1
                steps_since_checkp = 0
            else:
                steps_since_checkp += 1
        return checkpoints

    if (iters - 1) < WARMUP:
        return 0
    return ((iters - 1 - WARMUP) // cfreq) + 1


def _get_exec_throughput(input_file, baseline, cfreq):
    if not os.path.exists(input_file):
        return None

    exec_times = []
    extra_times = []
    checkpoint_counts = []
    exit_code = None

    with open(input_file, 'r') as f:
        for line in f.readlines():
            if 'EXECUTION TIME' in line:
                m = re.search(r'EXECUTION TIME:\s*([0-9]+(?:\.[0-9]+)?)', line)
                if m:
                    exec_times.append(float(m.group(1)))
            elif 'MMAP/UMAP' in line:
                tokens = line.split(" ")
                try:
                    extra_times.append(float(tokens[-2]) / 1000.0)
                except Exception:
                    pass
            elif 'Number of checkpoints:' in line:
                m = re.search(r'Number of checkpoints:\s*(\d+)', line)
                if m:
                    checkpoint_counts.append(int(m.group(1)))
            elif '[EXIT_CODE]' in line:
                try:
                    exit_code = int(line.strip().split()[-1])
                except Exception:
                    exit_code = None

    if exit_code is not None and exit_code != 0:
        print(f"[WARN] Skip failed run for {baseline}: {input_file} (EXIT_CODE={exit_code})")
        return None

    if len(exec_times) == 0:
        raise RuntimeError(
            f"No EXECUTION TIME found in {input_file}. "
            "Please make sure the run finished successfully."
        )

    expected_ckpt = _expected_checkpoints(cfreq, baseline)
    if expected_ckpt > 0 and len(checkpoint_counts) > 0:
        if any(c != expected_ckpt for c in checkpoint_counts):
            print(
                f"[WARN] Skip invalid run for {baseline}: {input_file} "
                f"(expected checkpoints={expected_ckpt}, parsed={checkpoint_counts})"
            )
            return None

    exec_time = max(exec_times)
    extra_time = max(extra_times) if len(extra_times) > 0 else 0.0
    exec_time -= extra_time

    if exec_time <= 0:
        raise RuntimeError(
            f"Invalid EXECUTION TIME parsed from {input_file}: {exec_time}. "
            "Please make sure the run finished successfully and log contains 'EXECUTION TIME'."
        )

    thr = (iters - WARMUP) / exec_time
    return thr


def collect():
    throughput_dict = {}
    throughput_list = []

    available_index = []
    # 当前吞吐统计只对比 PCcheck / MultiStream；CheckFreq 可忽略。
    for baseline in ["pccheck", "multistream"]:
        baseline_thr = []
        missing_logs = []
        for cf in cfreqs:
            input_file = f"{this_dir}/opt_27/log_opt27_{baseline}_{cf}.txt"
            thr = _get_exec_throughput(input_file, baseline, cf)
            if thr is None:
                missing_logs.append(input_file)
                continue
            baseline_thr.append(thr)

        # 该 baseline 的日志不完整时，跳过，避免后续 DataFrame/绘图维度错乱
        if len(missing_logs) > 0:
            print(f"[WARN] Skip baseline '{baseline}', missing logs:")
            for p in missing_logs:
                print(f"  - {p}")
            continue

        throughput_list.append(baseline_thr)
        throughput_dict[label_dict[baseline]] = baseline_thr
        available_index.append(label_dict[baseline])

    if len(throughput_list) == 0:
        raise RuntimeError(
            "No complete baseline logs found. "
            "Please run experiments first or adjust baselines in collect()."
        )

    print(throughput_list)
    column_header = [str(x) for x in cfreqs]
    df = pd.DataFrame(throughput_list, columns=column_header, index=available_index)
    df.to_csv(f'fig8_opt27.csv')
    return throughput_dict


def collect_bloom7b():
    throughput_dict = {}
    throughput_list = []

    available_index = []
    for baseline in ["pccheck", "multistream"]:
        baseline_thr = []
        missing_logs = []
        for cf in cfreqs:
            input_file = f"{this_dir}/bloom_7b/log_bloom7b_{baseline}_{cf}.txt"
            thr = _get_exec_throughput(input_file, baseline, cf)
            if thr is None:
                missing_logs.append(input_file)
                continue
            baseline_thr.append(thr)

        if len(missing_logs) > 0:
            print(f"[WARN] Skip baseline '{baseline}' (Bloom 7B), missing logs:")
            for p in missing_logs:
                print(f"  - {p}")
            continue

        throughput_list.append(baseline_thr)
        throughput_dict[label_dict[baseline]] = baseline_thr
        available_index.append(label_dict[baseline])

    if len(throughput_list) == 0:
        raise RuntimeError(
            "No complete Bloom 7B baseline logs found. "
            "Please run experiments first or adjust baselines in collect_bloom7b()."
        )

    print(throughput_list)
    column_header = [str(x) for x in cfreqs]
    df = pd.DataFrame(throughput_list, columns=column_header, index=available_index)
    df.to_csv(f'fig8_bloom7b.csv')
    return throughput_dict


def _plot_impl(data, output_filename):
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
    plt.savefig(output_filename, bbox_inches="tight", dpi=500, pad_inches=0.1)


def plot(data):
    _plot_impl(data, "fig8_opt27.png")


def plot_bloom7b(data):
    _plot_impl(data, "fig8_bloom7b.png")


if __name__ == "__main__":
    ip1 = "127.0.0.1"
    create_files(ip1)

    # OPT-2.7B experiments
    # run(ip1)
    # data = collect()
    # plot(data)

    # Bloom-7B experiments (pccheck + multistream)
    # run_bloom7b(ip1)
    data_bloom = collect_bloom7b()
    plot_bloom7b(data_bloom)
