import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/data/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/code/pccheck/checkpoint_eval/models/vision/"
lib_path_stream = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

iters = 300
model = "vgg16"
batchsize = 32
num_threads = 8

cfreqs = [0, 1, 5, 10, 15, 20, 25, 30]
max_async = [1, 2, 4, 6, 8]

# Multistream configuration (fixed layer groups, only vary async)
multistream_num_layer_groups = 6  # Fixed layer groups for multistream


# 1. Run PCCheck
def run():
    for num_conc in max_async:
        for cf in cfreqs:
            print(f"Run PCCheck with num_async {num_conc} cfreq {cf}")
            proc = f"python {script_dir}/train_pccheck.py --dataset imagenet  --batchsize {batchsize} --arch {model} --cfreq {cf} "\
                f"--bench_total_steps {iters} --max-async {num_conc} --num-threads {num_threads} --c_lib_path {lib_path} > log_{num_conc}_{cf}.txt 2>&1"
            os.system(proc)


# 1b. Run Multistream PCCheck (same structure as PCCheck, only vary async)
def run_multistream():
    for num_conc in max_async:
        for cf in cfreqs:
            print(f"Run Multistream with num_async {num_conc} cfreq {cf}")
            proc = f"python {script_dir}/train_multistream.py --dataset imagenet --batchsize {batchsize} --arch {model} --cfreq {cf} "\
                f"--bench_total_steps {iters} --max-async {num_conc} --num-threads {num_threads} --num_layer_groups {multistream_num_layer_groups} "\
                f"--c_lib_path {lib_path_stream} > log_multistream_{num_conc}_{cf}.txt 2>&1"
            os.system(proc)


# 2. collect measurements for PCCheck
def collect():

    def get_time(num_conc, cf):
        thr  = 0.0
        input_file = f"log_{num_conc}_{cf}.txt"
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    thr = float(tokens[-2])
                    break
        return thr

    slowdown_list = []
    for num_conc in max_async:
        thr_list_c = []
        for cf in cfreqs:
            thr_list_c.append(get_time(num_conc, cf))
        base_time = thr_list_c[0] if thr_list_c[0] > 0 else 1.0
        slowdown_list_c = [x/base_time for x in thr_list_c]
        slowdown_list.append(slowdown_list_c)

    column_header = [str(cf) for cf in cfreqs]
    index_header = [str(x) for x in max_async]
    df = pd.DataFrame(slowdown_list, columns = column_header, index = index_header)
    df.to_csv('fig12.csv')
    return slowdown_list


# 2b. collect measurements for Multistream (same structure as PCCheck)
def collect_multistream():

    def get_time(num_conc, cf):
        thr = 0.0
        input_file = f"log_multistream_{num_conc}_{cf}.txt"
        try:
            with open(input_file, 'r') as f:
                for line in f.readlines():
                    if 'EXECUTION TIME' in line:
                        tokens = line.split(" ")
                        thr = float(tokens[-2])
                        break
        except FileNotFoundError:
            print(f"Warning: {input_file} not found")
        return thr

    slowdown_list = []
    for num_conc in max_async:
        thr_list_c = []
        for cf in cfreqs:
            thr_list_c.append(get_time(num_conc, cf))
        # Avoid division by zero
        base_time = thr_list_c[0] if thr_list_c[0] > 0 else 1.0
        slowdown_list_c = [x/base_time for x in thr_list_c]
        slowdown_list.append(slowdown_list_c)
    
    # Save results
    column_header = [str(cf) for cf in cfreqs]
    index_header = [str(x) for x in max_async]
    df = pd.DataFrame(slowdown_list, columns=column_header, index=index_header)
    df.to_csv('fig12_multistream.csv')
    
    return slowdown_list


# 3. plot PCCheck async comparison
def plot(data):
    width = 0.15
    label_font_size = 27
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(cfreqs[1:]))
    bars = []
    async_checkp = [f"{x} async checkp" for x in max_async]

    for i,(d,l) in enumerate(zip(data, async_checkp)):

        print(d)
        bar = ax.bar(
            x + width * i, d[1:], width,
            label=l, #yerr=method2err[method],
            align='edge'
        )
        bars.append(bar)

    x_tick_positions = x + width * (len(async_checkp)/2)
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=cfreqs[1:], fontsize=label_font_size
    )
    ax.tick_params(axis='both', which='minor', labelsize=label_font_size)

    ax.set_yscale('log')
    ax.set_ylabel('Slowdown over \n no checkpointing', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint frequency', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', ncol=1, fontsize=24)
    #plt.title(model, fontsize=label_font_size)
    plt.savefig("fig12.png", bbox_inches="tight")


# 3b. plot comparison: PCCheck vs Multistream (side by side for each async config)
def plot_comparison(pccheck_data, multistream_data):
    """
    Plot comparison between PCCheck and Multistream for all async configs
    
    Args:
        pccheck_data: PCCheck slowdown data (list of lists, one per async level)
        multistream_data: Multistream slowdown data (same structure as pccheck_data)
    """
    label_font_size = 27
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(cfreqs[1:]))
    num_async = len(max_async)
    total_width = 0.8
    bar_width = total_width / (num_async * 2)  # PCCheck and Multistream for each async
    
    colors_pccheck = plt.cm.Blues(np.linspace(0.4, 0.9, num_async))
    colors_multistream = plt.cm.Purples(np.linspace(0.4, 0.9, num_async))
    
    for i, (num_conc, pc_data, ms_data) in enumerate(zip(max_async, pccheck_data, multistream_data)):
        offset = (i - num_async/2 + 0.5) * bar_width * 2
        # PCCheck bar
        ax.bar(x + offset, pc_data[1:], bar_width, 
               label=f'PCCheck {num_conc} async' if i == 0 else '', 
               color=colors_pccheck[i], edgecolor='navy', linewidth=0.5)
        # Multistream bar
        ax.bar(x + offset + bar_width, ms_data[1:], bar_width,
               label=f'Multistream {num_conc} async' if i == 0 else '',
               color=colors_multistream[i], edgecolor='purple', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(cfreqs[1:], fontsize=label_font_size)
    ax.tick_params(axis='y', labelsize=label_font_size)
    ax.set_yscale('log')
    ax.set_ylabel('Slowdown over \n no checkpointing', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint frequency', fontsize=label_font_size)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='PCCheck'),
                       Patch(facecolor='#9B59B6', label='Multistream')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=24)
    
    plt.tight_layout()
    plt.savefig("fig12_pccheck_vs_multistream.png", bbox_inches="tight")
    print("Saved comparison plot to fig12_pccheck_vs_multistream.png")


# 3c. plot Multistream async comparison (same format as PCCheck)
def plot_multistream(data):
    """Plot Multistream async comparison in the same format as PCCheck"""
    width = 0.15
    label_font_size = 27
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(cfreqs[1:]))
    bars = []
    async_checkp = [f"{x} async checkp" for x in max_async]

    for i, (d, l) in enumerate(zip(data, async_checkp)):
        print(d)
        bar = ax.bar(
            x + width * i, d[1:], width,
            label=l,
            align='edge'
        )
        bars.append(bar)

    x_tick_positions = x + width * (len(async_checkp)/2)
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=cfreqs[1:], fontsize=label_font_size
    )
    ax.tick_params(axis='both', which='minor', labelsize=label_font_size)

    ax.set_yscale('log')
    ax.set_ylabel('Slowdown over \n no checkpointing', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint frequency', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', ncol=1, fontsize=24)
    plt.savefig("fig12_multistream.png", bbox_inches="tight")
    print("Saved Multistream plot to fig12_multistream.png")


# 3d. plot both PCCheck and Multistream side by side (selected async)
def plot_side_by_side(pccheck_data, multistream_data, selected_async=2):
    """
    Plot PCCheck vs Multistream comparison for a specific async level
    """
    label_font_size = 24
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(cfreqs[1:]))
    width = 0.35
    
    # Find index for selected_async
    async_idx = max_async.index(selected_async)
    
    # Get data for selected async level
    pccheck_slowdown = pccheck_data[async_idx][1:]
    multistream_slowdown = multistream_data[async_idx][1:]
    
    # Plot bars
    bar1 = ax.bar(x - width/2, pccheck_slowdown, width, label=f'PCCheck', color='#3498db')
    bar2 = ax.bar(x + width/2, multistream_slowdown, width, label=f'Multistream', color='#9B59B6')
    
    ax.set_xlabel('Checkpoint frequency', fontsize=label_font_size)
    ax.set_ylabel('Slowdown over \n no checkpointing', fontsize=label_font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(cfreqs[1:], fontsize=label_font_size-4)
    ax.tick_params(axis='y', labelsize=label_font_size-4)
    ax.set_yscale('log')
    ax.legend(fontsize=label_font_size-4, loc='upper right')
    
    plt.title(f'{model.upper()} - PCCheck vs Multistream (async={selected_async})', fontsize=label_font_size)
    plt.tight_layout()
    plt.savefig(f"fig12_comparison_async{selected_async}.png", bbox_inches="tight")
    print(f"Saved comparison plot to fig12_comparison_async{selected_async}.png")


if __name__ == "__main__":
    parser_main = argparse.ArgumentParser(description="Run PCCheck async sensitivity analysis")
    parser_main.add_argument("--mode", type=str, default="all", 
                        choices=["all", "pccheck", "multistream", "collect", "plot"],
                        help="Run mode: all, pccheck only, multistream only, collect, or plot")
    parser_main.add_argument("--selected-async", type=int, default=2,
                        help="Async level for comparison plots (default: 2)")
    args_main = parser_main.parse_args()
    
    if args_main.mode == "all":
        # Run both PCCheck and Multistream
        # run()
        run_multistream()
        # Collect results
        pccheck_data = collect()
        multistream_data = collect_multistream()
        # Plot
        plot(pccheck_data)
        plot_multistream(multistream_data)
        plot_comparison(pccheck_data, multistream_data)
        plot_side_by_side(pccheck_data, multistream_data, selected_async=args_main.selected_async)
        
    elif args_main.mode == "pccheck":
        run()
        df = collect()
        plot(df)
        
    elif args_main.mode == "multistream":
        run_multistream()
        multistream_data = collect_multistream()
        plot_multistream(multistream_data)
        
    elif args_main.mode == "collect":
        pccheck_data = collect()
        multistream_data = collect_multistream()
        plot(pccheck_data)
        if multistream_data:
            plot_multistream(multistream_data)
            plot_comparison(pccheck_data, multistream_data)
            plot_side_by_side(pccheck_data, multistream_data, selected_async=args_main.selected_async)
        
    elif args_main.mode == "plot":
        # Read from CSV files
        pccheck_df = pd.read_csv('fig12.csv', index_col=0)
        pccheck_data = pccheck_df.values.tolist()
        plot(pccheck_data)
        
        # Try to read multistream data
        try:
            multistream_df = pd.read_csv('fig12_multistream.csv', index_col=0)
            multistream_data = multistream_df.values.tolist()
            plot_multistream(multistream_data)
            plot_comparison(pccheck_data, multistream_data)
            plot_side_by_side(pccheck_data, multistream_data, selected_async=args_main.selected_async)
        except FileNotFoundError:
            print("Warning: fig12_multistream.csv not found, skipping multistream plots")