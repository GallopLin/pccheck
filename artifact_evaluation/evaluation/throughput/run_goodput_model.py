import sys
import os
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

home_dir = os.path.expanduser("~")
default_pccheck_lib_path = f"{home_dir}/data/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
default_multistream_lib_path = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
lib_path = os.environ.get("PCCHECK_LIB_PATH", default_pccheck_lib_path)
lib_path_stream = os.environ.get("PCCHECK_LIB_PATH_STREAM", default_multistream_lib_path)

cfreqs = [0, 1, 10, 25, 50, 100]

model_scripts_dir = {
    "transformer": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch",
    "bert": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/BERT",
    "opt13": f"{home_dir}/code/transformers/examples/pytorch/language-modeling",
    "opt27": f"{home_dir}/code/transformers/examples/pytorch/language-modeling",
}

batch_size_dir = {"opt13": 1, "opt27": 1, "transformer": 64, "bert": 3}
iters_dir = {"opt13": 300, "opt27": 300, "transformer": 300, "bert": 300}
label_dict = {"cfreq": "CheckFreq", "gpm": "GPM", "pccheck": "PCcheck", "multistream": "Multistream"}

N_pccheck = {
    "transformer": 4,
    "opt13": 2,
    "bert": 4,
    "opt27": 2,
    "bloom7b": 2,
}

# Multistream uses layer groups for parallel writes.
# These values must match run_throughput_model.py.
N_multistream = {
    "transformer": 2,  # train_multistream.py in throughput runner
    "opt13": 6,        # run_clm_multistream.py in throughput runner
    "bert": 4,         # run_squad_multistream.py in throughput runner
    "opt27": 8,
    "bloom7b": 6,
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

METHOD_COLORS = {
    "CheckFreq": '#4392B8',
    "GPM": '#E27733',
    "PCcheck": '#A7B972',
    "CascadeSave": '#9B59B6',
    "Ideal": '#7f7f7f',
}
METHOD_MARKERS = {
    "CheckFreq": '*',
    "GPM": 's',
    "PCcheck": 'o',
    "CascadeSave": 'd',
}
METHOD_ORDER = ["CheckFreq", "GPM", "PCcheck", "CascadeSave", "Ideal"]
PAPER_MODEL_ORDER = ["transformer", "bert", "opt13", "opt27", "bloom7b"]
PAPER_MODEL_TITLES = {
    "transformer": "(a) Transformer-XL",
    "bert": "(b) BERT-Large",
    "opt13": "(c) OPT-1.3B",
    "opt27": "(d) OPT-2.7B",
    "bloom7b": "(e) BLOOM-7B",
}
SCRIPT_DIR = Path(__file__).resolve().parent
GPUS_TRACE_PATH = SCRIPT_DIR / "gpus_trace.csv"
PAPER_CFIG_INTERVALS = [0, 1, 10, 25, 50, 100]
H100_EFFECTIVE_ITER_OFFSET = 2
DERIVED_PAPER_LOG_DIRS = {
    "opt27": SCRIPT_DIR / "opt_27",
    "bloom7b": SCRIPT_DIR / "bloom_7b",
}
# Calibrate load time from the three models with measured loading logs.
# The checkpoint sizes below are parsed from their PCcheck logs and closely
# match the saved checkpoint payload loaded by loading.py.
LOAD_TIME_CALIBRATION = {
    "transformer": {"checkpoint_size_gb": 767801704 * 4 / 1e9, "load_time_sec": 1.9144635200500488},
    "bert": {"checkpoint_size_gb": 1338501128 * 4 / 1e9, "load_time_sec": 2.5099053382873535},
    "opt13": {"checkpoint_size_gb": 5468946432 * 4 / 1e9, "load_time_sec": 8.992764711380005},
}


def normalize_method_name(method_name):
    normalized = method_name.strip().lower()
    alias_map = {
        "checkfreq": "CheckFreq",
        "cfreq": "CheckFreq",
        "gpm": "GPM",
        "pccheck": "PCcheck",
        "multistream": "CascadeSave",
        "multistream ": "CascadeSave",
        "cascadesave": "CascadeSave",
        "ideal": "Ideal",
    }
    return alias_map.get(normalized, method_name.strip())


def normalize_plot_data(data):
    normalized_data = {}
    for method_name, method_values in data.items():
        canonical_name = normalize_method_name(method_name)
        normalized_data[canonical_name] = [float(x) for x in method_values]

    ordered_data = {}
    for method_name in METHOD_ORDER:
        if method_name in normalized_data:
            ordered_data[method_name] = normalized_data[method_name]
    for method_name, method_values in normalized_data.items():
        if method_name not in ordered_data:
            ordered_data[method_name] = method_values
    return ordered_data


def save_figure(fig, output_prefix, output_formats=("png", "pdf"), pad_inches=0.1):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for fmt in output_formats:
        output_path = output_prefix.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": pad_inches}
        if fmt == "png":
            save_kwargs["dpi"] = 500
        fig.savefig(output_path, **save_kwargs)


def build_legend_handles(method_names, line_width=1.6, marker_size=5):
    handles = []
    for method_name in method_names:
        if method_name == "Ideal":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=METHOD_COLORS["Ideal"],
                    linewidth=line_width,
                    linestyle="--",
                    label="Ideal",
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=METHOD_COLORS[method_name],
                    marker=METHOD_MARKERS[method_name],
                    linewidth=line_width,
                    markersize=marker_size,
                    label=method_name,
                )
            )
    return handles


def draw_goodput_plot(
    ax,
    intervals,
    data,
    label_font_size=36,
    title=None,
    show_xlabel=True,
    show_ylabel=True,
    line_width=2.0,
    marker_size=6,
    spine_width=0.6,
    tick_width=0.6,
):
    normalized_data = normalize_plot_data(data)
    x = np.arange(len(intervals[1:]), dtype=float)
    available_methods = [method_name for method_name in METHOD_ORDER if method_name in normalized_data]

    for method_name in available_methods:
        if method_name == "Ideal":
            continue
        ax.plot(
            x,
            normalized_data[method_name][1:],
            label=method_name,
            linewidth=line_width,
            marker=METHOD_MARKERS[method_name],
            markersize=marker_size,
            color=METHOD_COLORS[method_name],
        )

    if "Ideal" in normalized_data:
        ax.plot(
            x,
            normalized_data["Ideal"][1:],
            label="Ideal",
            linewidth=line_width,
            linestyle="--",
            color=METHOD_COLORS["Ideal"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in intervals[1:]], fontsize=label_font_size)
    ax.tick_params(axis="both", labelsize=label_font_size, width=tick_width, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if show_ylabel:
        ax.set_ylabel("Goodput (batches/sec)", fontsize=label_font_size)
    if show_xlabel:
        ax.set_xlabel("Checkpoint interval(iterations)", fontsize=label_font_size)
    if title is not None:
        ax.set_title(title, fontsize=label_font_size + 1, pad=10)

    y_max = max(max(method_values[1:]) for method_values in normalized_data.values())
    ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1.0)
    return available_methods


def load_plot_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    intervals = [int(column_name) for column_name in df.columns]
    data = {
        row_name: pd.to_numeric(df.loc[row_name], errors="coerce").fillna(0.0).tolist()
        for row_name in df.index
    }
    return intervals, normalize_plot_data(data)


def estimate_load_time_from_checkpoint_size(checkpoint_size_gb):
    sizes = np.array(
        [entry["checkpoint_size_gb"] for entry in LOAD_TIME_CALIBRATION.values()],
        dtype=float,
    )
    load_times = np.array(
        [entry["load_time_sec"] for entry in LOAD_TIME_CALIBRATION.values()],
        dtype=float,
    )
    slope, intercept = np.polyfit(sizes, load_times, 1)
    return max(0.0, float(slope * checkpoint_size_gb + intercept))


def parse_h100_throughput(
    log_path,
    effective_iter_offset=H100_EFFECTIVE_ITER_OFFSET,
    return_details=False,
):
    exec_times = []
    iter_counts = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "EXECUTION TIME:" in line:
                tokens = line.split()
                exec_times.append(float(tokens[-2]))
            elif "Number of iterations:" in line:
                try:
                    iter_counts.append(
                        int(line.split("Number of iterations:")[-1].split(",")[0].strip())
                    )
                except Exception:
                    pass

    if not exec_times:
        raise ValueError(f"No EXECUTION TIME found in {log_path}")
    if not iter_counts:
        raise ValueError(f"No iteration count found in {log_path}")

    exec_time = max(exec_times)
    raw_iters = max(iter_counts)
    effective_iters = max(0, raw_iters - effective_iter_offset)
    throughput = 0.0
    if exec_time > 0 and effective_iters > 0:
        throughput = effective_iters / exec_time

    if return_details:
        return throughput, {
            "raw_iterations": raw_iters,
            "effective_iterations": effective_iters,
            "execution_time_sec": exec_time,
            "effective_iter_offset": effective_iter_offset,
        }
    return throughput


def parse_average_write_time_seconds(log_path, method_name):
    values = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if method_name == "PCcheck":
                if "MSYNC TOOK" in line:
                    match = re.search(r"MSYNC TOOK\s+([0-9]*\.?[0-9]+)\s*ms", line)
                    if match:
                        values.append(float(match.group(1)) / 1000.0)
            elif method_name == "CascadeSave":
                if "All streams synced" in line and " ms" in line:
                    match = re.search(
                        r"All streams synced \(merged\)\s+[0-9]*\.?[0-9]+\s+GB\s+in\s+([0-9]*\.?[0-9]+)\s*ms",
                        line,
                    )
                    if match:
                        values.append(float(match.group(1)) / 1000.0)
            else:
                raise ValueError(f"Unsupported method_name: {method_name}")

    if not values:
        raise ValueError(f"No checkpoint write samples found in {log_path}")
    return float(sum(values) / len(values))


def parse_checkpoint_size_gb(log_path, method_name):
    sizes_gb = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if method_name == "PCcheck":
                if "total size is" in line:
                    tokens = line.split("total size is", 1)[1].split(",", 1)[0].strip()
                    sizes_gb.append(float(tokens) * 4 / 1e9)
            elif method_name == "CascadeSave":
                if "Total size (4 copies):" in line and "GB" in line:
                    size_token = line.rsplit("(", 1)[-1].split("GB", 1)[0].strip()
                    sizes_gb.append(float(size_token))
                elif "All streams synced (merged)" in line:
                    size_token = line.split("All streams synced (merged)", 1)[1].split("GB", 1)[0].strip()
                    if size_token:
                        sizes_gb.append(float(size_token))
            else:
                raise ValueError(f"Unsupported method_name: {method_name}")

    if not sizes_gb:
        raise ValueError(f"No checkpoint size found in {log_path}")

    # Multi-rank logs can report slightly different shard sizes per rank; use the
    # mean of unique values to estimate a representative reload cost.
    unique_sizes = sorted({round(size_gb, 6) for size_gb in sizes_gb})
    return float(sum(unique_sizes) / len(unique_sizes))


def build_goodput_from_h100_logs(model_name, log_dir, output_csv_path):
    num_fails, time_sec = get_fails_iters(pd.read_csv(GPUS_TRACE_PATH))
    intervals = PAPER_CFIG_INTERVALS
    method_specs = {
        "PCcheck": "pccheck",
        "CascadeSave": "multistream",
    }

    throughput_by_method = {}
    write_time_by_method = {}
    load_time_by_method = {}
    checkpoint_size_by_method = {}
    iteration_meta = {}

    for method_name, log_suffix in method_specs.items():
        throughput_values = []
        write_times = [0.0]
        iteration_meta[method_name] = {}
        for cfreq in intervals:
            log_path = log_dir / f"log_{model_name}_{log_suffix}_{cfreq}.txt"
            throughput, iter_detail = parse_h100_throughput(log_path, return_details=True)
            throughput_values.append(throughput)
            iteration_meta[method_name][cfreq] = iter_detail
            if cfreq != 0:
                write_times.append(parse_average_write_time_seconds(log_path, method_name))

        size_log_path = log_dir / f"log_{model_name}_{log_suffix}_1.txt"
        checkpoint_size_gb = parse_checkpoint_size_gb(size_log_path, method_name)
        checkpoint_size_by_method[method_name] = checkpoint_size_gb
        load_time_by_method[method_name] = estimate_load_time_from_checkpoint_size(checkpoint_size_gb)
        throughput_by_method[method_name] = throughput_values
        write_time_by_method[method_name] = write_times

    for method_name in method_specs:
        iter_summary = ", ".join(
            f"cf={cfreq}:raw{iteration_meta[method_name][cfreq]['raw_iterations']}"
            f"/eff{iteration_meta[method_name][cfreq]['effective_iterations']}"
            for cfreq in intervals
        )
        print(f"[{model_name}] {method_name} iter stats ({H100_EFFECTIVE_ITER_OFFSET} offset): {iter_summary}")

    if model_name == "bloom7b" and 1 in iteration_meta["PCcheck"] and 1 in iteration_meta["CascadeSave"]:
        pc_it = iteration_meta["PCcheck"][1]["raw_iterations"]
        cs_it = iteration_meta["CascadeSave"][1]["raw_iterations"]
        print(
            f"[bloom7b] cfreq=1 raw iterations: PCcheck={pc_it}, CascadeSave={cs_it}. "
            "Goodput uses per-log parsed iterations, not a shared fixed iteration count."
        )

    ideal_reference_baseline = max(
        throughput_by_method,
        key=lambda method_name: throughput_by_method[method_name][0],
    )
    ideal_reference_throughput = throughput_by_method[ideal_reference_baseline][0]
    ideal_reference_iter_time = (
        1 / ideal_reference_throughput
        if ideal_reference_throughput > 0
        else float("inf")
    )
    ideal_load_time = min(load_time_by_method.values())

    rows = {}
    for method_name in ["PCcheck", "CascadeSave"]:
        baseline_values = []
        no_checkpoint_throughput = throughput_by_method[method_name][0]
        time_no_checkp = (
            1 / no_checkpoint_throughput
            if no_checkpoint_throughput > 0
            else float("inf")
        )
        for index, cfreq in enumerate(intervals):
            checkpoint_throughput = throughput_by_method[method_name][index]
            avg_iter_time_checkp = (
                1 / checkpoint_throughput
                if checkpoint_throughput > 0
                else float("inf")
            )
            baseline_values.append(
                get_goodput_model_baseline(
                    method_name,
                    num_fails,
                    cfreq,
                    time_sec,
                    avg_iter_time_checkp,
                    load_time_by_method[method_name],
                    time_no_checkp,
                    model_name,
                    write_time_by_method[method_name][index],
                )
            )
        rows[method_name] = baseline_values

    rows["Ideal"] = [
        get_goodput_model_baseline(
            "Ideal",
            num_fails,
            cfreq,
            time_sec,
            ideal_reference_iter_time,
            ideal_load_time,
            ideal_reference_iter_time,
            model_name,
            0.0,
        )
        for cfreq in intervals
    ]

    df = pd.DataFrame(rows, index=intervals).T
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path)

    print(
        f"[{model_name}] Saved {output_csv_path} "
        f"(PCcheck load≈{load_time_by_method['PCcheck']:.2f}s from {checkpoint_size_by_method['PCcheck']:.2f}GB, "
        f"CascadeSave load≈{load_time_by_method['CascadeSave']:.2f}s from {checkpoint_size_by_method['CascadeSave']:.2f}GB)"
    )


def ensure_paper_fig9_csvs(data_dir):
    data_dir = Path(data_dir)
    for model_name, log_dir in DERIVED_PAPER_LOG_DIRS.items():
        if not log_dir.exists():
            continue
        build_goodput_from_h100_logs(
            model_name=model_name,
            log_dir=log_dir,
            output_csv_path=data_dir / f"fig9_{model_name}.csv",
        )


def get_paper_layout(available_models):
    if len(available_models) <= 3:
        layout = {
            model_name: (0, slice(index * 2, index * 2 + 2))
            for index, model_name in enumerate(available_models)
        }
        return 1, 2 * len(available_models), layout

    if len(available_models) == 5 and available_models == PAPER_MODEL_ORDER:
        layout = {
            "transformer": (0, slice(0, 2)),
            "bert": (0, slice(2, 4)),
            "opt13": (0, slice(4, 6)),
            "opt27": (1, slice(1, 3)),
            "bloom7b": (1, slice(3, 5)),
        }
        return 2, 6, layout

    top_count = int(np.ceil(len(available_models) / 2))
    bottom_count = len(available_models) - top_count
    cols = max(2 * top_count, 2 * bottom_count + 2)
    layout = {}
    for index, model_name in enumerate(available_models[:top_count]):
        layout[model_name] = (0, slice(index * 2, index * 2 + 2))
    bottom_start = 1 if bottom_count and cols > bottom_count * 2 else 0
    for index, model_name in enumerate(available_models[top_count:]):
        layout[model_name] = (1, slice(bottom_start + index * 2, bottom_start + index * 2 + 2))
    return 2, cols, layout

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
        model_name = "facebook/opt-1.3b" if model == "opt13" else "facebook/opt-2.7b"
        batch_size = batch_size_dir[model]
        iters = iters_dir[model]
        proc = f"python3.9 {script_dir}/run_clm_cfreq.py --model_name_or_path {model_name} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq 200 --bench_total_steps {iters}"
        os.system(proc)
        return checkpoint_file


    if model == "transformer":
        checkpoint_file = run_transformer()
    elif model == "bert":
        checkpoint_file = run_bert()
    elif model in ["opt13", "opt27"]:
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
    elif baseline in ["Multistream", "CascadeSave"]:
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
    raw_index_by_normalized = {
        normalize_method_name(row_name): row_name
        for row_name in iter_times_df.index
    }
    
    # Get actual cfreqs from CSV file columns to ensure consistency
    actual_cfreqs = [int(x) for x in iter_times_df.columns.tolist()]
    print(f"Using checkpoint frequencies from CSV: {actual_cfreqs}")
    
    # Check if Multistream is available in the CSV
    has_multistream = "CascadeSave" in raw_index_by_normalized
    
    if has_multistream:
        baseline_list = ["CheckFreq", "GPM", "PCcheck", "CascadeSave"]
    else:
        baseline_list = ["CheckFreq", "GPM", "PCcheck"]
        print("Note: Multistream data not found in CSV, skipping Multistream")

    for i, baseline in enumerate(baseline_list):
        raw_row_name = raw_index_by_normalized.get(baseline)
        if raw_row_name is not None:
            throughput = list(iter_times_df.loc[raw_row_name])
            iter_times[baseline] = [1 / x if x != 0 else float('inf') for x in throughput]
        else:
            print(f"Warning: {baseline} not found in CSV, using zeros")
            iter_times[baseline] = [0] * len(actual_cfreqs)

    # Match the throughput baseline rule: Ideal uses the best cfreq=0 throughput
    # among all available methods, not a hard-coded reference method.
    cfreq0_throughputs = {}
    for baseline in baseline_list:
        raw_row_name = raw_index_by_normalized.get(baseline)
        if raw_row_name is not None:
            cfreq0_throughputs[baseline] = float(iter_times_df.loc[raw_row_name].iloc[0])
        else:
            cfreq0_throughputs[baseline] = 0.0
    ideal_reference_baseline = max(cfreq0_throughputs, key=cfreq0_throughputs.get)
    ideal_reference_throughput = cfreq0_throughputs[ideal_reference_baseline]
    ideal_reference_iter_time = (
        1 / ideal_reference_throughput
        if ideal_reference_throughput > 0
        else float('inf')
    )
    print(
        "Ideal baseline reference (max cfreq=0 throughput): "
        f"{ideal_reference_baseline} = {ideal_reference_throughput:.4f}"
    )

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
            if baseline == "CascadeSave":
                tw = Tw_multistream_model[i] if i < len(Tw_multistream_model) else 0
            else:
                tw = Tw_pccheck_model[i] if i < len(Tw_pccheck_model) else 0
            
            if baseline == "Ideal":
                ref_iter_time = ideal_reference_iter_time
                ref_no_checkp_time = ideal_reference_iter_time
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


def plot_model(
    model,
    data,
    actual_cfreqs,
    output_prefix=None,
    output_formats=("png", "pdf"),
    label_font_size=36,
    figsize=(14, 7),
    legend_font_size=30,
    legend_loc="lower right",
    legend_bbox=None,
):
    if output_prefix is None:
        output_prefix = f"fig9_{model}"

    fig, ax = plt.subplots(figsize=figsize)
    available_methods = draw_goodput_plot(
        ax,
        actual_cfreqs,
        data,
        label_font_size=label_font_size,
        line_width=max(1.4, label_font_size / 18),
        marker_size=max(4.5, label_font_size / 6),
        spine_width=0.8,
        tick_width=0.8,
    )
    handles = build_legend_handles(
        [method_name for method_name in available_methods if method_name != "Ideal"] + (["Ideal"] if "Ideal" in available_methods else []),
        line_width=max(1.4, label_font_size / 18),
        marker_size=max(4.5, label_font_size / 6),
    )
    legend_kwargs = {
        "handles": handles,
        "loc": legend_loc,
        "ncol": 1 if len(available_methods) <= 4 else 2,
        "fontsize": legend_font_size,
        "frameon": False,
    }
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(**legend_kwargs)
    fig.tight_layout()
    save_figure(fig, output_prefix, output_formats=output_formats, pad_inches=0.1)
    plt.close(fig)


def plot_paper_fig9(
    data_dir,
    output_dir,
    combined_name="fig9_goodput_all",
    output_formats=("png", "pdf"),
    export_individual=True,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    ensure_paper_fig9_csvs(data_dir)
    available_models = [
        model_name
        for model_name in PAPER_MODEL_ORDER
        if (data_dir / f"fig9_{model_name}.csv").exists()
    ]
    if not available_models:
        raise FileNotFoundError(f"No fig9 CSV files found in {data_dir}")

    model_payload = {}
    for model_name in available_models:
        intervals, data = load_plot_csv(data_dir / f"fig9_{model_name}.csv")
        model_payload[model_name] = {"intervals": intervals, "data": data}
        if export_individual:
            plot_model(
                model_name,
                data,
                intervals,
                output_prefix=output_dir / f"fig9_{model_name}",
                output_formats=output_formats,
                label_font_size=22,
                figsize=(10, 5.8),
                legend_font_size=18,
                legend_loc="lower right",
            )

    rows, cols, layout = get_paper_layout(available_models)
    fig = plt.figure(figsize=(16.5, 4.8 if rows == 1 else 8.8))
    grid_spec = fig.add_gridspec(rows, cols, wspace=0.42 if rows == 1 else 0.48, hspace=0.78)

    for model_name in available_models:
        row_id, col_slice = layout[model_name]
        ax = fig.add_subplot(grid_spec[row_id, col_slice])
        draw_goodput_plot(
            ax,
            model_payload[model_name]["intervals"],
            model_payload[model_name]["data"],
            label_font_size=9,
            title=PAPER_MODEL_TITLES[model_name],
            show_xlabel=True,
            show_ylabel=True,
            line_width=0.9,
            marker_size=2.8,
            spine_width=0.5,
            tick_width=0.5,
        )

    legend_methods = []
    for method_name in METHOD_ORDER:
        if any(method_name in payload["data"] for payload in model_payload.values()):
            legend_methods.append(method_name)
    fig.legend(
        handles=build_legend_handles(legend_methods, line_width=1.2, marker_size=4.2),
        loc="upper center",
        ncol=len(legend_methods),
        fontsize=14,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    if rows == 1:
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.18, top=0.82, wspace=0.42)
    else:
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.86, wspace=0.48, hspace=0.78)
    save_figure(fig, output_dir / combined_name, output_formats=output_formats, pad_inches=0.05)
    plt.close(fig)

    print(f"Saved paper goodput figures to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run goodput simulation or generate paper-ready figures")
    parser.add_argument("model", nargs="?", type=str, choices=["transformer", "bert", "opt13", "opt27", "bloom7b"],
                        help="Model to simulate")
    parser.add_argument("--paper-data-dir", type=str, default=None,
                        help="Read fig9 CSV files from this directory and generate paper-ready PDF/PNG figures")
    parser.add_argument("--paper-output-dir", type=str, default=None,
                        help="Output directory for paper figures (default: sibling figures/ directory)")
    parser.add_argument("--paper-combined-name", type=str, default="fig9_goodput_all",
                        help="Output prefix for the combined paper figure")
    parser.add_argument("--paper-skip-individual", action="store_true",
                        help="Only generate the combined paper figure, skip per-model exports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.paper_data_dir is not None:
        data_dir = Path(args.paper_data_dir)
        output_dir = Path(args.paper_output_dir) if args.paper_output_dir else data_dir.parent / "figures"
        plot_paper_fig9(
            data_dir=data_dir,
            output_dir=output_dir,
            combined_name=args.paper_combined_name,
            export_individual=not args.paper_skip_individual,
        )
        sys.exit(0)

    if args.model is None:
        raise ValueError("model is required unless --paper-data-dir is specified")

    model = args.model
    if model in DERIVED_PAPER_LOG_DIRS and DERIVED_PAPER_LOG_DIRS[model].exists():
        output_csv_path = SCRIPT_DIR / f"fig9_{model}.csv"
        build_goodput_from_h100_logs(
            model_name=model,
            log_dir=DERIVED_PAPER_LOG_DIRS[model],
            output_csv_path=output_csv_path,
        )
        actual_cfreqs, data = load_plot_csv(output_csv_path)
    else:
        data, actual_cfreqs = get_goodput_model(model)
    print(data)
    plot_model(model, data, actual_cfreqs)
