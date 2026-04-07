from pathlib import Path
import itertools
import pandas as pd
from evaluation.metrics import mcnemar_test

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated"
TAG = "coverage_v2"

def latest_run(requested_model, dataset, strategy):
    matches = sorted(RAW_DIR.glob(f"{TAG}__{requested_model}__*__{dataset}__{strategy}__n*.csv"))
    return matches[-1] if matches else None

def main():
    agg = pd.read_csv(AGG_DIR / "coverage_v2_aggregated.csv")
    requested_models = list(dict.fromkeys(agg["requested_model_name"].tolist()))
    datasets = ["gsm8k","math_l1_l3","arc_challenge","truthfulqa_mc1"]
    strategies = ["zero_shot","cot","few_shot_cot"]

    rows = []
    for a, b in itertools.combinations(requested_models, 2):
        for d in datasets:
            for s in strategies:
                fa = latest_run(a, d, s)
                fb = latest_run(b, d, s)
                if fa is None or fb is None:
                    continue
                da = pd.read_csv(fa)[["sample_id","correct"]].rename(columns={"correct":"correct_a"})
                db = pd.read_csv(fb)[["sample_id","correct"]].rename(columns={"correct":"correct_b"})
                merged = da.merge(db, on="sample_id", how="inner")
                if merged.empty:
                    continue
                stats = mcnemar_test(merged["correct_a"], merged["correct_b"])
                rows.append({
                    "dataset": d,
                    "strategy": s,
                    "requested_model_a": a,
                    "requested_model_b": b,
                    "n_overlap": len(merged),
                    "acc_a": merged["correct_a"].mean(),
                    "acc_b": merged["correct_b"].mean(),
                    "n01_a0_b1": stats["n01"],
                    "n10_a1_b0": stats["n10"],
                    "mcnemar_stat": stats["statistic"],
                    "p_value": stats["p_value"],
                })

    out = pd.DataFrame(rows).sort_values(["dataset","strategy","requested_model_a","requested_model_b"])
    out.to_csv(AGG_DIR / "coverage_v2_pairwise_stats.csv", index=False)
    print(f"Saved: {AGG_DIR / 'coverage_v2_pairwise_stats.csv'}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
