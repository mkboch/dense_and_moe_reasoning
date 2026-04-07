from pathlib import Path
import itertools
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = ROOT / "results" / "aggregated"

RAW_PATH = AGG_DIR / "raw_concat_final.csv"
OUT_PATH = AGG_DIR / "pairwise_stats_final.csv"

def exact_two_sided_binom_pvalue(b: int, c: int) -> float:
    """
    Exact McNemar two-sided p-value via Binomial(n=b+c, p=0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def pmf(i: int) -> float:
        return math.comb(n, i) * (0.5 ** n)

    p_one_tail = sum(pmf(i) for i in range(0, k + 1))
    p_two = min(1.0, 2.0 * p_one_tail)
    return p_two

def mcnemar_cc_stat(b: int, c: int) -> float:
    """
    McNemar chi-square statistic with continuity correction.
    """
    denom = b + c
    if denom == 0:
        return 0.0
    return ((abs(b - c) - 1) ** 2) / denom

def pairwise_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    keys = ["dataset_name", "strategy"]
    for (dataset, strategy), g in df.groupby(keys):
        model_names = sorted(g["requested_model_name"].dropna().unique().tolist())

        for a, b in itertools.combinations(model_names, 2):
            ga = g[g["requested_model_name"] == a][["sample_id", "correct"]].copy()
            gb = g[g["requested_model_name"] == b][["sample_id", "correct"]].copy()

            ga = ga.rename(columns={"correct": "correct_a"})
            gb = gb.rename(columns={"correct": "correct_b"})

            merged = ga.merge(gb, on="sample_id", how="inner")
            if merged.empty:
                continue

            merged["correct_a"] = merged["correct_a"].fillna(0).astype(int)
            merged["correct_b"] = merged["correct_b"].fillna(0).astype(int)

            n_overlap = len(merged)
            acc_a = float(merged["correct_a"].mean())
            acc_b = float(merged["correct_b"].mean())

            # a wrong, b right
            n01 = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 1)).sum())
            # a right, b wrong
            n10 = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 0)).sum())

            stat = mcnemar_cc_stat(n01, n10)
            pval = exact_two_sided_binom_pvalue(n01, n10)

            rows.append({
                "dataset": dataset,
                "strategy": strategy,
                "requested_model_a": a,
                "requested_model_b": b,
                "n_overlap": n_overlap,
                "acc_a": acc_a,
                "acc_b": acc_b,
                "n01_a0_b1": n01,
                "n10_a1_b0": n10,
                "mcnemar_stat": stat,
                "p_value": pval,
            })

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "strategy", "requested_model_a", "requested_model_b"]
    ).reset_index(drop=True)

    return out

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    required = {"dataset_name", "strategy", "requested_model_name", "sample_id", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    pairwise = pairwise_stats(df)
    pairwise.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print("\n===== HEAD =====")
    print(pairwise.head(20).to_string(index=False))

    print("\n===== NaN CHECK =====")
    print(pairwise[["mcnemar_stat", "p_value"]].isna().sum().to_string())

    print("\n===== FILE CHECK =====")
    print(OUT_PATH.exists(), OUT_PATH.stat().st_size)

if __name__ == "__main__":
    main()
