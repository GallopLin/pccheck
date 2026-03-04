import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/data/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
lib_path_stream = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

# Default configurations (can be overridden by command line args)
DEFAULT_CFREQS = [0, 1, 10, 25, 50, 75, 100]
QUICK_CFREQS = [0, 1, 10, 25, 50, 100]  # Reduced set for quick testing

model_scripts_dir = {
    "transformer": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch",
    "bert": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/BERT",
    "opt13": f"{home_dir}/code/transformers/examples/pytorch/language-modeling",
    "opt27": f"{home_dir}/code/transformers/examples/pytorch/language-modeling"
}

# Default iterations
DEFAULT_ITERS = {"opt13": 300, "transformer": 350, "bert": 350, "opt27": 300}
QUICK_ITERS = {"opt13": 300, "transformer": 350, "bert": 350, "opt27": 300} # Reduced for quick testing (still needs warmup=50)

batch_size_dir = {"opt13": 1, "transformer": 64, "bert": 3, "opt27": 1}

label_dict = {
    "cfreq": "CheckFreq",
    "gpm": "GPM",
    "pccheck": "PCcheck",
    "multistream": "Multistream"
}

WARMUP = 50 # iterations - 大模型(OPT-2.7B)需要~50步让CUDA内存分配器和set_storage的瞬态开销稳定
# 注意：训练脚本中的 warmup 也必须与此值一致！
# 小模型(BERT/Transformer-XL)不需要这么多 warmup，但统一使用 50 简化逻辑，
# 相应地增加了它们的总迭代数(300→350)以保持足够的有效迭代。

# Global variables set by parse_args
cfreqs = DEFAULT_CFREQS
iters_dir = DEFAULT_ITERS
methods_to_run = None  # None means run all

def run_opt():
    global methods_to_run
    os.makedirs("opt13", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]

    # run cfreq
    if methods_to_run is None or "cfreq" in methods_to_run:
        print("Run for CheckFreq")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_cfreq.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > opt13/log_opt13_cfreq_{cf}.txt"
            os.system(proc)

    # run GPM
    if methods_to_run is None or "gpm" in methods_to_run:
        print("Run for GPM")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_gpm.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > opt13/log_opt13_gpm_{cf}.txt"
            os.system(proc)

    # run PCcheck
    if methods_to_run is None or "pccheck" in methods_to_run:
        print("Run for PCcheck")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_pccheck.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async 2 --num_threads 2 --psize 4 --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path} > opt13/log_opt13_pccheck_{cf}.txt"
            os.system(proc)

    # run Multistream PCcheck
    if methods_to_run is None or "multistream" in methods_to_run:
        print("Run for Multistream PCcheck")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_multistream.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async 2 --num_threads 2 --num_layer_groups 6 --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path_stream} > opt13/log_opt13_multistream_{cf}.txt 2>&1"
            os.system(proc)



def run_opt27():
    """Run OPT-2.7B benchmarks for all checkpoint methods.
    
    OPT-2.7B (facebook/opt-2.7b) has 32 transformer layers, ~2.7B parameters.
    Checkpoint size ~30GB (with optimizer states in FP32).
    Compared to OPT-1.3B:
      - 32 layers vs 24 layers
      - ~2x checkpoint size
      - num_layer_groups=8 (32 layers / 4 layers per group)
      - max_async=2, psize=8 (larger bucket for bigger model)
    """
    global methods_to_run
    os.makedirs("opt27", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]

    # run cfreq
    if methods_to_run is None or "cfreq" in methods_to_run:
        print("Run for CheckFreq")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_cfreq.py --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > opt27/log_opt27_cfreq_{cf}.txt"
            os.system(proc)

    # run GPM
    if methods_to_run is None or "gpm" in methods_to_run:
        print("Run for GPM")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_gpm.py --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > opt27/log_opt27_gpm_{cf}.txt"
            os.system(proc)

    # run PCcheck
    if methods_to_run is None or "pccheck" in methods_to_run:
        print("Run for PCcheck")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_pccheck.py --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async 2 --num_threads 2 --psize 8 --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path} > opt27/log_opt27_pccheck_{cf}.txt"
            os.system(proc)

    # run Multistream PCcheck
    if methods_to_run is None or "multistream" in methods_to_run:
        print("Run for Multistream PCcheck")
        for cf in cfreqs:
            os.system("rm -rf output")
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_clm_multistream.py --model_name_or_path facebook/opt-2.7b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async 2 --num_threads 2 --num_layer_groups 8 --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path_stream} > opt27/log_opt27_multistream_{cf}.txt 2>&1"
            os.system(proc)


def run_transformer():
    global methods_to_run
    os.makedirs("transformer", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]
    this_dir = f"{home_dir}/code/pccheck/artifact_evaluation/evaluation/throughput"

    # run cfreq
    if methods_to_run is None or "cfreq" in methods_to_run:
        print("Run for CheckFreq")
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"cd {script_dir} && python3.9 train_checkfreq.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > {this_dir}/transformer/log_transformer_cfreq_{cf}.txt"
            os.system(proc)

    # run gpm
    if methods_to_run is None or "gpm" in methods_to_run:
        print("Run for GPM")
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"cd {script_dir} && python3.9 train_gpm.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > {this_dir}/transformer/log_transformer_gpm_{cf}.txt"
            os.system(proc)

    # run pccheck
    if methods_to_run is None or "pccheck" in methods_to_run:
        print("Run for PCCheck")
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"cd {script_dir} && python3.9 train_pccheck.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} --max_async 4 --num_threads 2 --c_lib_path {lib_path} > {this_dir}/transformer/log_transformer_pccheck_{cf}.txt"
            os.system(proc)

    # run multistream pccheck
    if methods_to_run is None or "multistream" in methods_to_run:
        print("Run for Multistream PCCheck")
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"cd {script_dir} && python3.9 train_multistream.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} --max_async 4 --num_threads 2 --num_layer_groups 2 --c_lib_path {lib_path_stream} > {this_dir}/transformer/log_transformer_multistream_{cf}.txt 2>&1"
            os.system(proc)


def run_bert():
    global methods_to_run
    os.makedirs("bert", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]

    if methods_to_run is None or "cfreq" in methods_to_run:
        print("Run for CheckFreq")
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_squad_chfreq.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --cfreq {cf} --bench_total_steps {iters} --max_steps {iters} --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json > bert/log_bert_cfreq_{cf}.txt"
            os.system(proc)

    if methods_to_run is None or "gpm" in methods_to_run:
        print("Run for GPM")
        # run gpm
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_squad_gpm.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --cfreq {cf} --bench_total_steps {iters} --max_steps {iters} --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json > bert/log_bert_gpm_{cf}.txt"
            os.system(proc)

    if methods_to_run is None or "pccheck" in methods_to_run:
        print("Run for PCCheck")
        # run pccheck
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_squad_pccheck.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json --cfreq {cf} --max_async 4 --num_threads 2 --bench_total_steps {iters} --c_lib_path {lib_path} > bert/log_bert_pccheck_{cf}.txt"
            os.system(proc)

    if methods_to_run is None or "multistream" in methods_to_run:
        print("Run for Multistream PCCheck")
        # run multistream
        for cf in cfreqs:
            print(f"Checkpoint Frequency {cf}")
            proc = f"python3.9 {script_dir}/run_squad_multistream.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json --cfreq {cf} --max_async 4 --num_threads 2 --num_layer_groups 6 --bench_total_steps {iters} --c_lib_path {lib_path_stream} > bert/log_bert_multistream_{cf}.txt 2>&1"
            os.system(proc)


def run(model):
    if model == "transformer":
        run_transformer()
    elif model == "bert":
        run_bert()
    elif model == "opt13":
        run_opt()
    elif model == "opt27":
        run_opt27()
    else:
        raise NotImplementedError


def collect_model(model):
    def get_exec_throughput(input_file, baseline):
        exec_time = 0.0
        extra_time = 0.0
        iter_count = None
        try:
            with open(input_file, 'r') as f:
                for line in f.readlines():
                    if 'EXECUTION TIME' in line:
                        tokens = line.split(" ")
                        exec_time = float(tokens[-2])
                    elif 'MMAP/UMAP' in line:
                        tokens = line.split(" ")
                        extra_time = float(tokens[-2]) / 1000  # convert in sec
                    elif 'Number of iterations' in line:
                        # -- BENCHMARK ENDED: Total time: X sec, Number of iterations: Y, Number of checkpoints: Z
                        try:
                            iter_count = int(line.split('Number of iterations:')[-1].split(',')[0].strip())
                        except Exception:
                            pass
            # Only subtract MMAP/UMAP time for non-GPM methods.
            # For GPM, mmap/munmap is integral to its checkpoint mechanism
            # (GPU writes directly to mmap'd persistent memory), so subtracting
            # it would incorrectly remove checkpoint overhead from the measurement.
            # Other methods (PCcheck, CheckFreq, Multistream) don't output this
            # line, so extra_time is 0 for them anyway.
            if baseline != "gpm":
                exec_time -= extra_time
            # EXECUTION TIME 从 warmup 结束后开始计时，因此需要从迭代次数中扣除 warmup。
            effective_iters = iters_dir[model]
            if iter_count is not None:
                effective_iters = max(0, iter_count - WARMUP)
            if exec_time <= 0 or effective_iters <= 0:
                return 0.0
            thr = effective_iters / exec_time
            return thr
        except Exception as e:
            print(f"Warning: Failed to read {input_file}: {e}")
            return 0.0

    throughput_dict = {}
    throughput_list = []

    # All models now support all 4 methods including multistream
    baselines = ["cfreq", "gpm", "pccheck", "multistream"]
    index_header = ["CheckFreq", "GPM", "PCcheck", "Multistream"]

    for baseline in baselines:
        baseline_thr = []
        for cf in cfreqs:
            input_file = f"{model}/log_{model}_{baseline}_{cf}.txt"
            thr = get_exec_throughput(input_file, baseline)
            baseline_thr.append(thr)
        throughput_list.append(baseline_thr)
        throughput_dict[label_dict[baseline]] = baseline_thr

    print(throughput_list)
    column_header = [str(x) for x in cfreqs]
    df = pd.DataFrame(throughput_list, columns = column_header, index = index_header)
    df.to_csv(f'fig8_{model}.csv')
    return throughput_dict


def plot_model(model, data):

    # All models now use 4 methods including Multistream
    colors = ['#4392B8', '#E27733', '#A7B972', '#9B59B6']  # 4 methods
    width = 0.18
    
    label_font_size = 36
    fig, ax = plt.subplots(figsize=(16, 8))  # Wider figure for 4 methods
    x = np.arange(len(cfreqs[1:]))
    bars = []

    print(data)
    for method_id, (method_key, method_data) in enumerate(data.items()):

        # slowdown = [baseline_throughput[0]/x for x in data[method]]
        # print(method, slowdown)
        bar = ax.bar(
            x + width * method_id, method_data[1:], width,
            label=method_key,
            align='edge',
            color=colors[method_id]
        )
        bars.append(bar)

    plt.yticks(fontsize=label_font_size)

    # ✅ 修复：使用所有方法 cfreq=0 吞吐量的最大值作为 baseline
    # 这更公平，因为不同脚本的 cfreq=0 行为可能略有不同
    cfreq0_throughputs = [method_data[0] for method_data in data.values()]
    baseline_thr = max(cfreq0_throughputs)
    print(f"Baseline throughput (max of cfreq=0): {baseline_thr:.4f}")
    print(f"  All cfreq=0 values: {dict(zip(data.keys(), cfreq0_throughputs))}")
    
    # Adjust position based on number of methods
    num_methods = len(data)
    ax.plot(x + width * num_methods / 2, [baseline_thr]*len(x), color='black', marker="s", linewidth=3, markersize=8, label='No Checkpoint')

    x_tick_positions = x + width * num_methods / 2
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=cfreqs[1:], fontsize=label_font_size,
    )
    plt.yticks(fontsize=label_font_size)

    ax.set_ylabel('Throughput (iterations/sec)', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint interval(iterations)', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()

    ncol = min(5, num_methods + 1)  # +1 for "No Checkpoint"
    plt.legend(handles, labels, loc='upper left', ncol=ncol, fontsize=label_font_size-4, bbox_to_anchor=(-0.025, 1.2))
    plt.savefig(f"fig8_{model}.png", bbox_inches="tight", dpi=500, pad_inches=0.1)


def parse_args():
    parser = argparse.ArgumentParser(description='Run throughput benchmarks for checkpoint methods')
    parser.add_argument('model', type=str, choices=['transformer', 'bert', 'opt13', 'opt27'],
                        help='Model to benchmark')
    parser.add_argument('--quick', action='store_false',
                        help='Quick mode: fewer iterations (50) and fewer checkpoint frequencies [0, 10, 50]')
    parser.add_argument('--iters', type=int, default=None,
                        help='Number of iterations per run (overrides default)')
    parser.add_argument('--cfreqs', type=str, default=None,
                        help='Comma-separated checkpoint frequencies, e.g., "0,10,50"')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods to run: cfreq,gpm,pccheck,multistream (default: all)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip running benchmarks, only collect and plot existing results')
    parser.add_argument('--only-run', action='store_true',
                        help='Only run benchmarks, skip collect and plot')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = args.model
    
    # Configure based on arguments
    if args.quick:
        cfreqs = QUICK_CFREQS
        iters_dir = QUICK_ITERS
        print(f"🚀 Quick mode: iters={iters_dir[model]}, cfreqs={cfreqs}")
    else:
        cfreqs = DEFAULT_CFREQS
        iters_dir = DEFAULT_ITERS
    
    # Override with specific values if provided
    if args.iters is not None:
        iters_dir = {k: args.iters for k in iters_dir}
        print(f"📝 Using custom iterations: {args.iters}")
    
    if args.cfreqs is not None:
        cfreqs = [int(x) for x in args.cfreqs.split(',')]
        print(f"📝 Using custom checkpoint frequencies: {cfreqs}")
    
    if args.methods is not None:
        methods_to_run = args.methods.split(',')
        print(f"📝 Running only methods: {methods_to_run}")
    
    # Estimate execution time
    num_methods = len(methods_to_run) if methods_to_run else 4
    num_cfreqs = len(cfreqs)
    total_runs = num_methods * num_cfreqs
    est_time_per_run = iters_dir[model] * 0.5  # rough estimate: 0.5 sec per iteration
    est_total_time = total_runs * est_time_per_run / 60  # in minutes
    print(f"⏱️  Estimated time: ~{est_total_time:.1f} minutes ({total_runs} runs × {iters_dir[model]} iters)")
    
    if not args.skip_run:
        run(model)
    
    if not args.only_run:
        data = collect_model(model)
        plot_model(model, data)