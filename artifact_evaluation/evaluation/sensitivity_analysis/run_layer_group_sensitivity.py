"""
Layer Group Sensitivity Analysis for MultiStream Checkpoint.

Sweeps num_layer_groups across multiple models and checkpoint frequencies,
collects throughput data, fits an analytical model for optimal K selection,
and generates publication-quality figures.

Usage:
    # Run experiments for all models
    python3.9 run_layer_group_sensitivity.py --mode run

    # Collect results from logs
    python3.9 run_layer_group_sensitivity.py --mode collect

    # Plot figures and fit analytical model
    python3.9 run_layer_group_sensitivity.py --mode plot

    # Run a single model
    python3.9 run_layer_group_sensitivity.py --mode run --model bert

    # Full pipeline
    python3.9 run_layer_group_sensitivity.py --mode all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

home_dir = os.path.expanduser("~")
lib_path_stream = f"{home_dir}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "layer_group_sensitivity")

MODEL_CONFIG = {
    "bert": {
        "script_dir": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/BERT",
        "num_transformer_layers": 24,
        # layers_per_group_sweep: g must evenly divide num_transformer_layers (L=24)
        # divisors of 24: 1, 2, 3, 4, 6, 8, 12, 24  =>  K = 24, 12, 8, 6, 4, 3, 2, 1
        "layers_per_group_sweep": [2, 4, 6, 8, 12],
        "cfreqs": [10, 25, 50],
        "iters": 250,
        "batch_size": 3,
        "max_async": 4,
        "num_threads": 2,
        "checkpoint_size_gb": 4.0,
        "display_name": "BERT-Large",
    },
    "opt13": {
        "script_dir": f"{home_dir}/code/transformers/examples/pytorch/language-modeling",
        "num_transformer_layers": 24,
        # divisors of 24: 1, 2, 3, 4, 6, 8, 12, 24  =>  K = 24, 12, 8, 6, 4, 3, 2, 1
        "layers_per_group_sweep": [2, 4, 6, 8, 12],
        "cfreqs": [10, 25, 50],
        "iters": 250,
        "batch_size": 1,
        "max_async": 2,
        "num_threads": 2,
        "checkpoint_size_gb": 16.2,
        "display_name": "OPT-1.3B",
    },
    "transformer": {
        "script_dir": f"{home_dir}/code/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch",
        "num_transformer_layers": 16,
        # divisors of 16: 1, 2, 4, 8, 16  =>  K = 16, 8, 4, 2, 1
        "layers_per_group_sweep": [2, 4, 8],
        "cfreqs": [10, 25, 50],
        "iters": 250,
        "batch_size": 64,
        "max_async": 4,
        "num_threads": 2,
        "checkpoint_size_gb": 2.7,
        "display_name": "Transformer-XL",
    },
}

WARMUP = 3


# ============================================================
# 1. Run experiments
# ============================================================

def _build_cmd_bert(cfg, K, cf):
    sd = cfg["script_dir"]
    return (
        f"python3.9 {sd}/run_squad_multistream.py "
        f"--bert_model=bert-large-uncased "
        f"--train_batch_size {cfg['batch_size']} "
        f"--output_dir output "
        f"--vocab_file {sd}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt "
        f"--config_file {sd}/bert_configs/large.json "
        f"--do_train "
        f"--train_file {sd}/download/squad/v1.1/train-v1.1.json "
        f"--cfreq {cf} "
        f"--max_async {cfg['max_async']} "
        f"--num_threads {cfg['num_threads']} "
        f"--num_layer_groups {K} "
        f"--bench_total_steps {cfg['iters']} "
        f"--c_lib_path {lib_path_stream}"
    )


def _build_cmd_opt13(cfg, K, cf):
    sd = cfg["script_dir"]
    return (
        f"python3.9 {sd}/run_clm_multistream.py "
        f"--model_name_or_path facebook/opt-1.3b "
        f"--output_dir output "
        f"--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 "
        f"--do_train "
        f"--max_async {cfg['max_async']} "
        f"--num_threads {cfg['num_threads']} "
        f"--num_layer_groups {K} "
        f"--per_device_train_batch_size {cfg['batch_size']} "
        f"--cfreq {cf} "
        f"--bench_total_steps {cfg['iters']} "
        f"--overwrite_output_dir "
        f"--c_lib_path {lib_path_stream}"
    )


def _build_cmd_transformer(cfg, K, cf):
    sd = cfg["script_dir"]
    return (
        f"cd {sd} && python3.9 train_multistream.py "
        f"--config_file wt103_base.yaml "
        f"--batch_size {cfg['batch_size']} "
        f"--cfreq {cf} "
        f"--bench_total_steps {cfg['iters']} "
        f"--max_async {cfg['max_async']} "
        f"--num_threads {cfg['num_threads']} "
        f"--num_layer_groups {K} "
        f"--c_lib_path {lib_path_stream}"
    )


CMD_BUILDERS = {
    "bert": _build_cmd_bert,
    "opt13": _build_cmd_opt13,
    "transformer": _build_cmd_transformer,
}


def _log_path(model, g, cf):
    return os.path.join(OUTPUT_DIR, model, f"log_{model}_g{g}_cf{cf}.txt")


def run_experiments(models):
    for model in models:
        cfg = MODEL_CONFIG[model]
        L = cfg["num_transformer_layers"]
        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)
        builder = CMD_BUILDERS[model]

        for g in cfg["layers_per_group_sweep"]:
            assert L % g == 0, f"g={g} does not evenly divide L={L} for model {model}"
            K = L // g
            for cf in cfg["cfreqs"]:
                log = _log_path(model, g, cf)
                if os.path.exists(log):
                    print(f"[SKIP] {log} exists")
                    continue
                print(f"[RUN] {model}  g={g} (K={K})  cfreq={cf}")
                cmd = builder(cfg, K, cf)
                os.system(f"{cmd} > {log} 2>&1")


# ============================================================
# 2. Collect results
# ============================================================

def _parse_throughput(log_file, total_iters):
    exec_time = 0.0
    iter_count = None
    try:
        with open(log_file, 'r', errors='replace') as f:
            for line in f:
                if 'EXECUTION TIME' in line:
                    tokens = line.split()
                    exec_time = float(tokens[-2])
                elif 'Number of iterations' in line:
                    try:
                        iter_count = int(
                            line.split('Number of iterations:')[-1].split(',')[0].strip()
                        )
                    except Exception:
                        pass
    except FileNotFoundError:
        return None

    effective_iters = total_iters
    if iter_count is not None:
        effective_iters = max(0, iter_count - WARMUP)
    if exec_time <= 0 or effective_iters <= 0:
        return None
    return effective_iters / exec_time


def collect_results(models):
    all_results = {}
    for model in models:
        cfg = MODEL_CONFIG[model]
        L = cfg["num_transformer_layers"]
        rows = []
        for g in cfg["layers_per_group_sweep"]:
            K = L // g
            row = {"g": g, "K": K}
            for cf in cfg["cfreqs"]:
                log = _log_path(model, g, cf)
                thr = _parse_throughput(log, cfg["iters"])
                row[f"cf{cf}"] = thr
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("g").reset_index(drop=True)
        csv_path = os.path.join(OUTPUT_DIR, f"layer_group_{model}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[SAVED] {csv_path}")
        print(df.to_string(index=False))
        print()
        all_results[model] = df
    return all_results


# ============================================================
# 3. Analytical model fitting
# ============================================================

def throughput_model(K, T_base, alpha, beta):
    """
    Analytical throughput model:
        Thr(K) = T_base / (1 + alpha/K + beta*K)

    - T_base : ideal throughput (no checkpoint overhead)
    - alpha/K : residual pipeline stall — decreases as K grows
                (more overlap between save and backward)
    - beta*K  : per-group scheduling overhead — increases with K
                (CUDA stream launches, event sync, small-block I/O)

    Optimal K* = sqrt(alpha / beta)
    """
    return T_base / (1.0 + alpha / K + beta * K)


def fit_model(K_vals, thr_vals):
    """Fit the analytical model and return (params, K_opt)."""
    K_arr = np.array(K_vals, dtype=float)
    thr_arr = np.array(thr_vals, dtype=float)

    mask = np.isfinite(thr_arr) & (thr_arr > 0)
    if mask.sum() < 3:
        return None, None

    K_arr = K_arr[mask]
    thr_arr = thr_arr[mask]

    try:
        p0 = [thr_arr.max(), 1.0, 0.001]
        popt, _ = curve_fit(
            throughput_model, K_arr, thr_arr, p0=p0,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        T_base, alpha, beta = popt
        K_opt = np.sqrt(alpha / beta) if beta > 0 else K_arr[np.argmax(thr_arr)]
        return popt, K_opt
    except Exception as e:
        print(f"  [WARN] curve_fit failed: {e}")
        return None, None


# ============================================================
# 4. Plotting
# ============================================================

def plot_sensitivity(all_results):
    """
    Generate two figures:
      fig13a: throughput vs K for each model (one subplot per cfreq)
      fig13b: normalized throughput vs K with fitted curve
    """
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

    plot_cfreqs = [10, 25]

    # ---- Figure 13a: multi-panel, one per model ----
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False)
    axes = axes[0]

    colors_cf = {10: '#E74C3C', 25: '#3498DB', 50: '#2ECC71'}
    markers_cf = {10: 'o', 25: 's', 50: '^'}

    for idx, (model, df) in enumerate(all_results.items()):
        ax = axes[idx]
        cfg = MODEL_CONFIG[model]
        L = cfg["num_transformer_layers"]
        g_vals = df["g"].values
        K_vals = df["K"].values  # K = L // g, used for fitting

        for cf in plot_cfreqs:
            col = f"cf{cf}"
            if col not in df.columns:
                continue
            thr_vals = df[col].values

            base_col = "cf0"
            base_thr = None
            if base_col in df.columns:
                base_thr = df[base_col].dropna().max()
            if base_thr and base_thr > 0:
                norm_thr = thr_vals / base_thr
            else:
                norm_thr = thr_vals

            valid = np.isfinite(norm_thr) & (norm_thr > 0)
            ax.plot(
                g_vals[valid], norm_thr[valid],
                marker=markers_cf.get(cf, 'D'),
                color=colors_cf.get(cf, 'gray'),
                linewidth=2, markersize=8,
                label=f'interval={cf}',
            )

            # Fit analytical model in K-space, then convert x-axis back to g
            popt, K_opt = fit_model(K_vals, thr_vals)
            if popt is not None:
                K_fine = np.linspace(K_vals.min(), K_vals.max(), 200)
                g_fine = L / K_fine  # convert K back to g for x-axis
                thr_fine = throughput_model(K_fine, *popt)
                if base_thr and base_thr > 0:
                    thr_fine_norm = thr_fine / base_thr
                else:
                    thr_fine_norm = thr_fine
                ax.plot(g_fine, thr_fine_norm, '--',
                        color=colors_cf.get(cf, 'gray'), alpha=0.5, linewidth=1.5)

                K_opt_clamp = max(K_vals.min(), min(K_opt, K_vals.max()))
                g_opt = L / K_opt_clamp
                ax.axvline(x=g_opt, color=colors_cf.get(cf, 'gray'),
                           linestyle=':', alpha=0.6, linewidth=1)

        ax.set_xlabel('Layers per group ($g$)', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Normalized throughput', fontsize=14)
        ax.set_title(cfg["display_name"], fontsize=16)
        ax.set_xticks(g_vals)
        # Secondary tick labels show corresponding K = L/g
        ax.set_xticklabels([f"{g}\n(K={L//g})" for g in g_vals], fontsize=9)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path_a = os.path.join(OUTPUT_DIR, "figures", "fig13_layer_group_sensitivity.png")
    plt.savefig(path_a, dpi=300, bbox_inches='tight')
    plt.savefig(path_a.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"[SAVED] {path_a}")
    plt.close()

    # ---- Figure 13b: fitting summary table (print to console) ----
    print("\n" + "=" * 80)
    print("Analytical Model Fitting: Thr(K) = T_base / (1 + alpha/K + beta*K)")
    print("Optimal K* = sqrt(alpha / beta)")
    print("=" * 80)

    fit_results = []
    for model, df in all_results.items():
        cfg = MODEL_CONFIG[model]
        L = cfg["num_transformer_layers"]
        for cf in plot_cfreqs:
            col = f"cf{cf}"
            if col not in df.columns:
                continue
            K_vals = df["K"].values
            thr_vals = df[col].values
            popt, K_opt = fit_model(K_vals, thr_vals)
            if popt is not None:
                T_base, alpha, beta = popt
                g_opt = L / K_opt  # convert optimal K back to layers-per-group
                chunk_size_mb = cfg["checkpoint_size_gb"] * 1024 / (K_opt * 4)
                fit_results.append({
                    "Model": cfg["display_name"],
                    "Interval": cf,
                    "T_base": f"{T_base:.4f}",
                    "alpha": f"{alpha:.4f}",
                    "beta": f"{beta:.6f}",
                    "K*": f"{K_opt:.1f}",
                    "g* (layers/group)": f"{g_opt:.1f}",
                    "g* (rounded)": int(round(g_opt)),
                    "Chunk/stream (MB)": f"{chunk_size_mb:.0f}",
                })
                print(f"  {cfg['display_name']:16s}  interval={cf:3d}  "
                      f"T_base={T_base:.4f}  alpha={alpha:.4f}  beta={beta:.6f}  "
                      f"K*={K_opt:.1f}  g*={g_opt:.1f}  chunk/stream={chunk_size_mb:.0f}MB")

    if fit_results:
        fit_df = pd.DataFrame(fit_results)
        fit_csv = os.path.join(OUTPUT_DIR, "fitting_results.csv")
        fit_df.to_csv(fit_csv, index=False)
        print(f"\n[SAVED] {fit_csv}")

    # ---- Figure 13c: bar chart of optimal K vs actual K ----
    if fit_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        models_plotted = []
        k_opt_vals = []
        k_actual_vals = []

        # actual_g: layers-per-group used in the paper (g=4 for all models)
        actual_g = {"BERT-Large": 4, "OPT-1.3B": 4, "Transformer-XL": 4}

        for r in fit_results:
            key = (r["Model"], r["Interval"])
            if key not in [(r2["Model"], r2["Interval"]) for r2 in models_plotted]:
                models_plotted.append(r)
                k_opt_vals.append(float(r["g* (layers/group)"]))
                k_actual_vals.append(actual_g.get(r["Model"], 4))

        x = np.arange(len(models_plotted))
        labels = [f"{r['Model']}\n(int={r['Interval']})" for r in models_plotted]
        width = 0.35

        ax.bar(x - width / 2, k_actual_vals, width, label='Adopted $g$ (paper)', color='#3498DB')
        ax.bar(x + width / 2, k_opt_vals, width, label='Optimal $g^*$', color='#E74C3C')

        ax.set_ylabel('Layers per group ($g$)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path_c = os.path.join(OUTPUT_DIR, "figures", "fig13_optimal_K.png")
        plt.savefig(path_c, dpi=300, bbox_inches='tight')
        plt.savefig(path_c.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"[SAVED] {path_c}")
        plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer group sensitivity analysis for MultiStream checkpoint"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "run", "collect", "plot"],
        help="Execution mode",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_CONFIG.keys()),
        help="Run only a specific model (default: all)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated models to run, e.g. bert,opt13",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        invalid = [m for m in models if m not in MODEL_CONFIG]
        if invalid:
            raise ValueError(
                f"Invalid model(s): {invalid}. Valid choices: {list(MODEL_CONFIG.keys())}"
            )
    else:
        models = [args.model] if args.model else list(MODEL_CONFIG.keys())

    if args.mode in ("all", "run"):
        run_experiments(models)

    if args.mode in ("all", "collect"):
        collect_results(models)

    if args.mode in ("all", "plot"):
        all_results = {}
        for m in models:
            csv_path = os.path.join(OUTPUT_DIR, f"layer_group_{m}.csv")
            if os.path.exists(csv_path):
                all_results[m] = pd.read_csv(csv_path)
            else:
                print(f"[WARN] {csv_path} not found, skipping {m}")
        if all_results:
            plot_sensitivity(all_results)
        else:
            print("[ERROR] No data to plot")


if __name__ == "__main__":
    main()
