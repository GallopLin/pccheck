import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
cfreqs = [0, 1, 5, 10, 15, 20, 25, 30]
max_async = [1, 2, 4, 6, 8]
multistream_layer_groups = [4, 8, 16]
model = "vgg16"

def plot_consolidated(pccheck_data, multistream_data):
    """
    Plot all async levels in a single figure with subplots.
    Compares PCCheck vs MultiStream (all available groups).
    """
    num_plots = len(max_async)
    cols = 2
    rows = (num_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows), constrained_layout=True)
    axes = axes.flatten()
    
    x = np.arange(len(cfreqs[1:]))
    
    # Determine bar width based on how many groups we have
    available_groups = [g for g in multistream_layer_groups if g in multistream_data]
    num_bars = 1 + len(available_groups) # PCCheck + MS groups
    width = 0.8 / num_bars
    
    colors = {
        'pccheck': '#3498db', # Blue
        4: '#2ecc71',         # Green
        8: '#9B59B6',         # Purple
        16: '#e74c3c'         # Red
    }

    for i, async_val in enumerate(max_async):
        ax = axes[i]
        async_idx = i
        
        # PCCheck Data
        pccheck_vals = np.array(pccheck_data[async_idx][1:])
        pccheck_vals = np.where(pccheck_vals <= 0, np.nan, pccheck_vals)
        
        # Calculate offset for first bar (PCCheck)
        # We want to center the group of bars on x
        offset_base = - (num_bars - 1) / 2.0
        
        # Plot PCCheck
        ax.bar(x + (offset_base * width), pccheck_vals, width, label='PCCheck', color=colors['pccheck'])
        
        # Plot MultiStream Groups
        for j, group in enumerate(available_groups):
            ms_vals = np.array(multistream_data[group][async_idx][1:])
            ms_vals = np.where(ms_vals <= 0, np.nan, ms_vals)
            
            offset = offset_base + (j + 1)
            ax.bar(x + (offset * width), ms_vals, width, label=f'MultiStream ({group} grps)', color=colors.get(group, 'gray'))

        ax.set_title(f'Async Level = {async_val}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(cfreqs[1:])
        ax.set_yscale('log')
        ax.set_ylabel('Slowdown')
        ax.set_xlabel('Checkpoint Frequency')
        ax.grid(True, which="both", ls="-", alpha=0.1)
        
        # Only add legend to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc='upper right')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f'{model.upper()} - PCCheck vs MultiStream Consolidated Comparison', fontsize=18)
    output_file = 'fig12_consolidated.png'
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Saved consolidated plot to {output_file}")

def main():
    # Read PCCheck data
    try:
        pccheck_df = pd.read_csv('fig12.csv', index_col=0)
        pccheck_data = pccheck_df.values.tolist()
    except FileNotFoundError:
        print("fig12.csv not found")
        return

    # Read Multistream data
    multistream_results = {}
    for num_groups in multistream_layer_groups:
        try:
            df = pd.read_csv(f'fig12_multistream_{num_groups}groups.csv', index_col=0)
            multistream_results[num_groups] = df.values.tolist()
        except FileNotFoundError:
            pass
            
    if not multistream_results:
        print("No multistream data found")
        return

    # Generate consolidated plot
    plot_consolidated(pccheck_data, multistream_results)

if __name__ == "__main__":
    main()
