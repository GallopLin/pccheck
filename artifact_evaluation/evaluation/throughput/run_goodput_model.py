import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
lib_path_stream = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

cfreqs = [0, 1, 10, 25, 50, 100]

model_scripts_dir = {
    "transformer": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch",
    "bert": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/BERT",
    "opt13": f"{home_dir}/code/transformers/examples/pytorch/language-modeling",
}

batch_size_dir = {"opt13": 1, "transformer": 64, "bert": 3}
iters_dir = {"opt13": 300, "transformer": 300, "bert": 300}
label_dict = {"cfreq": "CheckFreq", "gpm": "GPM", "pccheck": "PCcheck", "multistream": "Multistream"}

N_pccheck = {
    "transformer": 4,
    "opt13": 2,
    "bert": 4,
    "opt_27": 2,
    "bloom_7": 2,
}

# Multistream uses layer groups for parallel writes
# Should match the actual num_layer_groups parameter used in training scripts
N_multistream = {
    "transformer": 6,  # num_layer_groups=6 in train_multistream.py
    "opt13": 6,
    "bert": 6,         # num_layer_groups=6 in run_squad_multistream.py
    "opt_27": 6,
    "bloom_7": 6,
}

colors = {
    "CheckFreq": '#4392B8',
    "GPM" : '#E27733',
    "PCcheck": '#A7B972',
    "Multistream": '#9B59B6'
}

markers = {
    "CheckFreq": '*',
    "GPM" : 's',
    "PCcheck": 'o',
    "Multistream": 'd'
}

def get_load_time(model):

    checkpoint_file = ""


    def run_transformer():
        script_dir = model_scripts_dir[model]
        checkpoint_file = f"{script_dir}/checkpoint-0-0.chk"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        proc = f"cd {script_dir} && python3.9 train_checkfreq.py --config_file wt103_base.yaml --batch_size 64 --cfreq 200 --bench_total_steps 200"
        os.system(proc)
        return checkpoint_file


    def run_bert():
        script_dir = model_scripts_dir[model]
        checkpoint_file = f"checkpoint-0-0.chk"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        proc = f"python3.9 {script_dir}/run_squad_chfreq.py --bert_model=bert-large-uncased --train_batch_size 3 --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --cfreq 200 --bench_total_steps 200 --max_steps 200 --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json"
        os.system(proc)
        return checkpoint_file


    def run_opt():
        script_dir = model_scripts_dir[model]
        checkpoint_file = f"checkpoint-0-0.chk"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        os.system("rm -rf output")
        proc = f"python3.9 {script_dir}/run_clm_cfreq.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size 1 --cfreq 200 --bench_total_steps 200"
        os.system(proc)
        return checkpoint_file


    if model == "transformer":
        checkpoint_file = run_transformer()
    elif model == "bert":
        checkpoint_file = run_bert()
    elif model == "opt13":
        checkpoint_file = run_opt()
    else:
        raise NotImplementedError

    # 2. Load and get time

    if model=="transformer":
        script_dir = model_scripts_dir[model]
        this_dir = f"{home_dir}/code/pccheck/artifact_evaluation/evaluation/throughput"
        os.system(f"cp loading.py {script_dir}/")
        os.system(f"cd {script_dir} && python3.9 loading.py checkpoint-0-0.chk > {this_dir}/loading_log_{model}.txt")
    else:
        os.system(f"python3.9 loading.py {checkpoint_file} > loading_log_{model}.txt")

    # 3. Read file and return
    with open(f"loading_log_{model}.txt", 'r') as f:
        for line in f.readlines():
            if "Time is" in line:
                tokens = line.split(" ")
                load_time = float(tokens[-2])

    return load_time


def get_fails_iters(trace_file):
    data = list(trace_file["Time"].iloc[[0, -1]])
    time_sec = data[1] - data[0]

    num_fails = 0
    prev_cores = 0
    for it, row in trace_file.iterrows():
        if it >= 0:
            if row["GPUs"] != prev_cores:
                num_fails += abs(row["GPUs"] - prev_cores) // 4
        prev_cores = row["GPUs"]
    return num_fails, time_sec


def get_time_redo(baseline, cfreq, time_no_checkp, loading_time, model, Tw_pccheck):
    if baseline in ["CheckFreq", "Gemini"]:
        time_redo = cfreq * time_no_checkp + loading_time
    elif baseline == "GPM":
        time_redo = cfreq * time_no_checkp / 2 + loading_time
    elif baseline == "PCcheck":
        time_redo = loading_time + cfreq * time_no_checkp / 2
        time_redo += (
            time_no_checkp
            * min(Tw_pccheck / time_no_checkp, cfreq * N_pccheck[model])
            / 2
        )
    elif baseline == "Multistream":
        # Multistream has better parallelism with 4 streams and layer-wise saving
        # Recovery time is similar to PCcheck but with more efficient writes
        time_redo = loading_time + cfreq * time_no_checkp / 2
        time_redo += (
            time_no_checkp
            * min(Tw_pccheck / time_no_checkp, cfreq * N_multistream[model])
            / 2
        )
    elif baseline == "Ideal":
        time_redo = cfreq * time_no_checkp / 2 + loading_time

    return time_redo


def get_goodput_model_baseline(
    baseline,
    num_fails,
    cfreq,
    total_time,
    avg_iter_time_checkp,
    loading_time,
    time_no_checkp,
    model,
    Tw_pccheck
):

    time_redo = get_time_redo(baseline, cfreq, time_no_checkp, loading_time, model, Tw_pccheck)

    time_redo_all = time_redo * num_fails
    # time_redo_all = 0
    time_rem = total_time - time_redo_all
    seen_batches = time_rem / avg_iter_time_checkp
    throughput = seen_batches / total_time
    # return seen_batches
    return max(0, throughput)


def get_goodput_model(model):

    load_time = get_load_time(model)
    print(load_time)
    num_fails, time_sec = get_fails_iters(pd.read_csv("gpus_trace.csv"))
    print(num_fails, time_sec)

    iter_times = {}
    iter_times_df = pd.read_csv(f"fig8_{model}.csv", header=0, index_col=0)
    
    # Get actual cfreqs from CSV file columns to ensure consistency
    actual_cfreqs = [int(x) for x in iter_times_df.columns.tolist()]
    print(f"Using checkpoint frequencies from CSV: {actual_cfreqs}")
    
    # Check if Multistream is available in the CSV
    has_multistream = "Multistream" in iter_times_df.index
    
    if has_multistream:
        baseline_list = ["CheckFreq", "GPM", "PCcheck", "Multistream"]
    else:
        baseline_list = ["CheckFreq", "GPM", "PCcheck"]
        print("Note: Multistream data not found in CSV, skipping Multistream")

    for i, baseline in enumerate(baseline_list):
        if baseline in iter_times_df.index:
            throughput = list(iter_times_df.loc[baseline])
            iter_times[baseline] = [1 / x if x != 0 else float('inf') for x in throughput]
        else:
            print(f"Warning: {baseline} not found in CSV, using zeros")
            iter_times[baseline] = [0] * len(actual_cfreqs)

    # Read Tw for PCcheck - parse "MSYNC TOOK xxx ms" format
    Tw_pccheck_model = [0]
    for cf in actual_cfreqs[1:]:
        input_file = f"{model}/log_{model}_pccheck_{cf}.txt"
        msync_times = []
        try:
            with open(input_file, 'r') as f:
                for line in f.readlines():
                    # PCcheck format: "MSYNC TOOK 1698.729813 ms"
                    if "MSYNC TOOK" in line:
                        tokens = line.split()
                        msync_times.append(float(tokens[2]) / 1000.0)  # convert ms to seconds
                    # Legacy format: "average is xxx"
                    elif "average is" in line:
                        tokens = line.split(" ")
                        msync_times.append(float(tokens[-1]))
            # Use average of all MSYNC times
            average_time = sum(msync_times) / len(msync_times) if msync_times else 0.0
        except FileNotFoundError:
            print(f"Warning: File not found: {input_file}, using default average_time=0")
            average_time = 0.0
        except Exception as e:
            print(f"Warning: Error reading {input_file}: {e}, using default average_time=0")
            average_time = 0.0
        Tw_pccheck_model.append(average_time)
    print(f"PCcheck Tw values: {Tw_pccheck_model}")

    # Read Tw for Multistream - parse "All streams synced (merged) ... in xxx ms" format
    Tw_multistream_model = [0]
    if has_multistream:
        for cf in actual_cfreqs[1:]:
            input_file = f"{model}/log_{model}_multistream_{cf}.txt"
            sync_times = []
            try:
                with open(input_file, 'r') as f:
                    for line in f.readlines():
                        # Multistream format: "All streams synced (merged) 3.07 GB in 1004.88 ms (3.06 GB/s)"
                        if "All streams synced" in line and " ms " in line:
                            # Extract time before "ms"
                            import re
                            match = re.search(r'in\s+([\d.]+)\s*ms', line)
                            if match:
                                sync_times.append(float(match.group(1)) / 1000.0)  # convert ms to seconds
                # Use average of all sync times
                average_time = sum(sync_times) / len(sync_times) if sync_times else 0.0
            except FileNotFoundError:
                # Fallback to PCcheck Tw if multistream log not found
                idx = len(Tw_multistream_model)
                average_time = Tw_pccheck_model[idx] if idx < len(Tw_pccheck_model) else 0.0
                print(f"Warning: Multistream log not found for cfreq={cf}, using PCcheck Tw={average_time}")
            except Exception as e:
                average_time = 0.0
                print(f"Warning: Error reading multistream log: {e}")
            Tw_multistream_model.append(average_time)
        print(f"Multistream Tw values: {Tw_multistream_model}")

    goodputs_dict = {}
    goodputs_list = []
    baseline_list_with_ideal = baseline_list + ["Ideal"]

    for baseline in baseline_list_with_ideal:
        goodput_baseline = []
        for i, cf in enumerate(actual_cfreqs):
            # Use appropriate Tw for each baseline
            if baseline == "Multistream":
                tw = Tw_multistream_model[i] if i < len(Tw_multistream_model) else 0
            else:
                tw = Tw_pccheck_model[i] if i < len(Tw_pccheck_model) else 0
            
            # Use Multistream as reference for Ideal if available, otherwise use PCcheck
            if baseline == "Ideal":
                if has_multistream:
                    ref_iter_time = iter_times["Multistream"][0]
                else:
                    ref_iter_time = iter_times["PCcheck"][0]
                ref_no_checkp_time = ref_iter_time
            else:
                ref_iter_time = iter_times[baseline][i]
                ref_no_checkp_time = iter_times[baseline][0]
            
            goodput_cf = get_goodput_model_baseline(
                baseline,
                num_fails,
                cf,
                time_sec,
                ref_iter_time,
                load_time,
                ref_no_checkp_time,
                model,
                tw
            )
            goodput_baseline.append(goodput_cf)
        goodputs_list.append(goodput_baseline)
        goodputs_dict[baseline] = goodput_baseline

    column_header = [str(x) for x in actual_cfreqs]
    index_header = baseline_list_with_ideal
    df = pd.DataFrame(goodputs_list, columns = column_header, index = index_header)
    df.to_csv(f'fig9_{model}.csv')
    return goodputs_dict, actual_cfreqs


def plot_model(model, data, actual_cfreqs):
    x = range(len(actual_cfreqs[1:]))
    label_font_size = 36
    fig, ax = plt.subplots(figsize=(14, 7))

    for method_id, (method_key, method_data) in enumerate(data.items()):
        print(method_key, method_data)
        if method_key == "Ideal":
            continue
        # Check if this method has color and marker defined
        if method_key in colors and method_key in markers:
            plt.plot(x, method_data[1:], label=method_key, linewidth=3,
                 marker=markers[method_key], markersize=10, color=colors[method_key])
        else:
            # Fallback for unknown methods
            plt.plot(x, method_data[1:], label=method_key, linewidth=3, markersize=10)

    plt.plot(x, data["Ideal"][1:], label='Ideal',
             linewidth=3, linestyle='--', color='grey')

    ax.set_ylabel('Goodput (batches/sec)', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint interval(iterations)', fontsize=label_font_size)

    plt.yticks(fontsize=label_font_size)
    plt.xticks(x, actual_cfreqs[1:], fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    # Adjust legend columns based on number of methods
    ncol = 1 if len(data) <= 4 else 2
    plt.legend(handles, labels, loc='lower right', ncol=ncol, fontsize=30)
    plt.savefig(f"fig9_{model}.png", bbox_inches="tight", dpi=500, pad_inches=0.1)



if __name__ == "__main__":
    model = sys.argv[1]
    data, actual_cfreqs = get_goodput_model(model)
    print(data)
    plot_model(model, data, actual_cfreqs)