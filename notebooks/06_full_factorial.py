# %% [markdown]
# # Week 6 — Full Factorial Experiment (120 Runs)
#
# Design: 4 methods × 3 similarity conditions × 10 seeds = 120 runs
#
# Session management strategy:
#   - Results saved as individual .pkl files after EVERY run
#   - check_completion() scans saved files so you resume exactly where you left off
#   - Run 20-25 experiments per Colab session (each ≈ 3-5 min on T4)
#   - Aim for 4-5 sessions total across the week
#
# ⚠ Before running: confirm ewc_lambda and er_buffer_size are set correctly
#   in DEFAULT_CONFIG (from Weeks 2 and 3).

# %% — Setup (paste into Colab cell 1 of every session)
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# Mount Drive at the START of every session
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/drive/MyDrive/continual-learning-similarity')

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.experiment import (run_experiment, save_result, load_all_results,
                             check_completion, DEFAULT_CONFIG)

Path("../results/raw").mkdir(parents=True, exist_ok=True)

# ── Load best hyperparameters from Weeks 2 and 3 ─────────────────────────────
config = DEFAULT_CONFIG.copy()

try:
    with open("../results/ewc_best_lambda.json") as f:
        config["ewc_lambda"] = json.load(f)["ewc_lambda"]
    print(f"EWC lambda loaded: {config['ewc_lambda']}")
except FileNotFoundError:
    print(f"Using default EWC lambda: {config['ewc_lambda']}  (run Week 2 first)")

try:
    with open("../results/er_best_config.json") as f:
        config["er_buffer_size"] = json.load(f)["er_buffer_size"]
    print(f"ER buffer size loaded: {config['er_buffer_size']}")
except FileNotFoundError:
    print(f"Using default ER buffer: {config['er_buffer_size']}  (run Week 3 first)")

config["checkpoint_dir"] = "../results/raw"
config["data_dir"]       = "../data"

# %% — Define the full factorial
METHODS      = ["finetune", "ewc", "er", "joint"]
SIMILARITIES = ["low", "medium", "high"]
SEEDS        = list(range(10))   # seeds 0-9

print(f"\nFull factorial: {len(METHODS)} methods × {len(SIMILARITIES)} "
      f"conditions × {len(SEEDS)} seeds = {len(METHODS)*len(SIMILARITIES)*len(SEEDS)} runs")

# %% — Check what's already done (run this at the START of every Colab session)
remaining = check_completion(METHODS, SIMILARITIES, SEEDS,
                              checkpoint_dir="../results/raw")
print(f"\nRun this cell at the start of each session to see what remains.")

# %% — MAIN EXPERIMENT LOOP
# IMPORTANT: adjust BATCH_SIZE to run 20-25 experiments before Colab timeout.
# After timeout: re-run the "Setup" and "Check" cells, then re-run this cell.
# Already-completed runs are skipped automatically.

BATCH_SIZE   = 25       # experiments per session (adjust as needed)
VERBOSE      = True

remaining = check_completion(METHODS, SIMILARITIES, SEEDS,
                              checkpoint_dir="../results/raw")

if not remaining:
    print("✓ All 120 experiments complete!")
else:
    batch = remaining[:BATCH_SIZE]
    print(f"\nRunning {len(batch)} experiments this session ...")
    print("(Interrupt and re-run to continue from checkpoint)\n")

    session_start = time.time()
    for i, (method, similarity, seed) in enumerate(batch):
        run_start = time.time()
        print(f"[{i+1}/{len(batch)}] {method:10s} | {similarity:6s} | seed={seed:02d} ... ",
              end="", flush=True)

        try:
            result = run_experiment(method, similarity, seed, config=config)
            path   = save_result(result, checkpoint_dir="../results/raw")
            elapsed = time.time() - run_start
            print(f"BWT={result['bwt']:+.4f}  AvgAcc={result['avg_accuracy']:.4f}  "
                  f"({elapsed:.0f}s)  → {os.path.basename(path)}")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    total = time.time() - session_start
    print(f"\nSession complete: {len(batch)} runs in {total/60:.1f} min")
    remaining_after = check_completion(METHODS, SIMILARITIES, SEEDS,
                                       checkpoint_dir="../results/raw")
    if not remaining_after:
        print("✓ ALL 120 EXPERIMENTS COMPLETE")

# %% — Build summary table (run after all 120 are done)
results = load_all_results("../results/raw")
print(f"\nLoaded {len(results)} results")

if len(results) < 120:
    print(f"⚠ Only {len(results)}/120 complete. Run the loop above to finish.")
else:
    rows = []
    for r in results:
        cka_val = r.get("cka_values", {}).get("task0_to_final", np.nan)
        rows.append({
            "method":     r["method"],
            "similarity": r["similarity"],
            "seed":       r["seed"],
            "bwt":        r["bwt"],
            "avg_accuracy": r["avg_accuracy"],
            "cka_task0_to_final": cka_val,
        })

    df = pd.DataFrame(rows)

    # Summary: mean ± std per (method, similarity) cell
    summary = df.groupby(["method", "similarity"]).agg(
        BWT_mean=("bwt",          "mean"),
        BWT_std =("bwt",          "std"),
        Acc_mean=("avg_accuracy",  "mean"),
        Acc_std =("avg_accuracy",  "std"),
        CKA_mean=("cka_task0_to_final", "mean"),
        CKA_std =("cka_task0_to_final", "std"),
    ).reset_index()

    print("\n" + "=" * 80)
    print("SUMMARY TABLE (mean ± std across 10 seeds)")
    print("=" * 80)
    print(summary.to_string(index=False))

    summary.to_csv("../results/bwt_summary_table.csv", index=False)
    print("\nSaved: results/bwt_summary_table.csv")

# %% — Two-way ANOVA: Method × Similarity on BWT
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

if len(results) == 120:
    print("\n" + "=" * 60)
    print("TWO-WAY ANOVA: BWT ~ Method × Similarity")
    print("=" * 60)

    df_anova = df.copy()
    # Encode as categorical for statsmodels
    df_anova["method"]     = pd.Categorical(df_anova["method"])
    df_anova["similarity"] = pd.Categorical(df_anova["similarity"],
                                             categories=["low","medium","high"],
                                             ordered=True)

    model = smf.ols("bwt ~ C(method) + C(similarity) + C(method):C(similarity)",
                    data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table.to_string())

    with open("../results/interaction_anova.txt", "w") as f:
        f.write("Two-Way ANOVA: BWT ~ Method × Similarity\n\n")
        f.write(anova_table.to_string())
    print("\nSaved: results/interaction_anova.txt")

    # Extract interaction term
    interaction_row = anova_table.loc[
        [r for r in anova_table.index if "method" in r and "similarity" in r][0]
    ]
    print(f"\n{'='*60}")
    print(f"HEADLINE RESULT — Interaction term:")
    print(f"  F = {interaction_row['F']:.4f}")
    print(f"  p = {interaction_row['PR(>F)']:.4f}")
    sig = "SIGNIFICANT" if interaction_row['PR(>F)'] < 0.05 else "NOT SIGNIFICANT"
    print(f"  → Interaction is {sig}")
    print(f"{'='*60}")
    print("\nRECORD THE FULL ANOVA TABLE IN YOUR LAB NOTEBOOK.")
