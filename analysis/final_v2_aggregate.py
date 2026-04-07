from pathlib import Path
import pandas as pd
from evaluation.metrics import accuracy_with_ci, compute_flops_per_token

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated"
TAG = "coverage_v2"

def main():
    files = sorted(RAW_DIR.glob(f"{TAG}__*__*__*__*__n*.csv"))
    files = [f for f in files if "__ALL_RUNS__" not in f.name]
    if not files:
        raise RuntimeError("No coverage_v2 per-run CSV files found.")

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rows = []
    group_cols = [
        "requested_model_name", "actual_model_name", "model_pretty_name", "hf_id", "architecture",
        "total_params_b", "active_params_b", "load_mode", "dataset_name", "strategy"
    ]

    for keys, g in raw.groupby(group_cols, dropna=False):
        stats = accuracy_with_ci(g["correct"])
        rows.append({
            "requested_model_name": keys[0],
            "actual_model_name": keys[1],
            "model_pretty_name": keys[2],
            "hf_id": keys[3],
            "architecture": keys[4],
            "total_params_b": keys[5],
            "active_params_b": keys[6],
            "load_mode": keys[7],
            "dataset_name": keys[8],
            "strategy": keys[9],
            "n": stats["n"],
            "num_correct": stats["num_correct"],
            "accuracy": stats["accuracy"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "mean_latency_sec": g["latency_sec"].mean(),
            "std_latency_sec": g["latency_sec"].std(ddof=0),
            "mean_output_tokens": g["n_output_tokens"].mean(),
            "mean_tokens_per_sec": g["tokens_per_sec"].mean(),
            "peak_vram_gb": g["peak_vram_gb"].max(),
            "flops_per_token": compute_flops_per_token(float(keys[6])),
        })

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).sort_values(["dataset_name", "requested_model_name", "strategy"])
    out.to_csv(AGG_DIR / "coverage_v2_aggregated.csv", index=False)

    weights = {"gsm8k":0.40, "math_l1_l3":0.30, "arc_challenge":0.20, "truthfulqa_mc1":0.10}
    out["dataset_weight"] = out["dataset_name"].map(weights)

    ws = []
    for (req_model, actual_model, strategy), g in out.groupby(["requested_model_name","actual_model_name","strategy"], dropna=False):
        ws.append({
            "requested_model_name": req_model,
            "actual_model_name": actual_model,
            "strategy": strategy,
            "weighted_accuracy": (g["accuracy"] * g["dataset_weight"]).sum(),
            "mean_latency_sec_across_tasks": g["mean_latency_sec"].mean(),
            "mean_vram_gb_across_tasks": g["peak_vram_gb"].mean(),
            "mean_flops_per_token": g["flops_per_token"].mean(),
        })
    pd.DataFrame(ws).sort_values(["requested_model_name","weighted_accuracy"], ascending=[True, False]).to_csv(
        AGG_DIR / "coverage_v2_weighted_summary.csv", index=False
    )

    print(f"Saved: {AGG_DIR / 'coverage_v2_aggregated.csv'}")
    print(f"Saved: {AGG_DIR / 'coverage_v2_weighted_summary.csv'}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
