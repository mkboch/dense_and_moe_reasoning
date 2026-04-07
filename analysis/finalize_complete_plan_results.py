from pathlib import Path
import itertools
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated"
FIG_DIR = ROOT / "figures"
FINAL_TAG = "complete_plan_v2"

DATASET_WEIGHTS = {
    "gsm8k": 0.40,
    "math_l1_l3": 0.30,
    "arc_challenge": 0.20,
    "truthfulqa_mc1": 0.10,
}

def ensure_dirs():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_final_raw():
    files = sorted(RAW_DIR.glob(f"{FINAL_TAG}__*.csv"))
    files = [
        f for f in files
        if "__ALL_RUNS__" not in f.name
        and "protocol_fix_smoke" not in f.name
    ]
    if not files:
        raise RuntimeError(f"No per-run raw CSVs found for tag={FINAL_TAG}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    if "correct" in df.columns:
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    if "latency_sec" in df.columns:
        df["latency_sec"] = pd.to_numeric(df["latency_sec"], errors="coerce")
    if "n_output_tokens" in df.columns:
        df["n_output_tokens"] = pd.to_numeric(df["n_output_tokens"], errors="coerce")
    if "tokens_per_sec" in df.columns:
        df["tokens_per_sec"] = pd.to_numeric(df["tokens_per_sec"], errors="coerce")
    if "peak_vram_gb" in df.columns:
        df["peak_vram_gb"] = pd.to_numeric(df["peak_vram_gb"], errors="coerce")
    if "total_params_b" in df.columns:
        df["total_params_b"] = pd.to_numeric(df["total_params_b"], errors="coerce")
    if "active_params_b" in df.columns:
        df["active_params_b"] = pd.to_numeric(df["active_params_b"], errors="coerce")
    return df

def wilson_ci_from_binary(correct_series, z=1.96):
    s = pd.Series(correct_series).fillna(0).astype(int)
    n = int(len(s))
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = float(s.mean())
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return phat, low, high

def compute_flops_per_token(active_params_b):
    if pd.isna(active_params_b):
        return float("nan")
    return float(active_params_b) * 2e9

def aggregate(df):
    rows = []
    group_cols = [
        "requested_model_name",
        "actual_model_name",
        "model_pretty_name",
        "hf_id",
        "architecture",
        "total_params_b",
        "active_params_b",
        "dataset_name",
        "strategy",
    ]
    for keys, g in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        correct_series = g["correct"].astype(int)
        acc, ci_low, ci_high = wilson_ci_from_binary(correct_series)
        rec.update({
            "n": int(len(g)),
            "num_correct": int(correct_series.sum()),
            "accuracy": acc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_latency_sec": float(g["latency_sec"].mean()),
            "std_latency_sec": float(g["latency_sec"].std(ddof=0)),
            "mean_output_tokens": float(g["n_output_tokens"].mean()),
            "mean_tokens_per_sec": float(g["tokens_per_sec"].mean()),
            "peak_vram_gb": float(g["peak_vram_gb"].max()),
            "flops_per_token": compute_flops_per_token(rec["active_params_b"]),
        })
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(
        ["dataset_name", "strategy", "requested_model_name"]
    ).reset_index(drop=True)
    return out

def weighted_summary(agg):
    rows = []
    for (req_model, act_model, strat), g in agg.groupby(
        ["requested_model_name", "actual_model_name", "strategy"], dropna=False
    ):
        weighted_acc = 0.0
        present_weight = 0.0
        for _, r in g.iterrows():
            w = DATASET_WEIGHTS.get(r["dataset_name"], 0.0)
            weighted_acc += w * float(r["accuracy"])
            present_weight += w
        if present_weight > 0:
            weighted_acc /= present_weight

        rows.append({
            "requested_model_name": req_model,
            "actual_model_name": act_model,
            "strategy": strat,
            "weighted_accuracy": weighted_acc,
            "mean_latency_sec_across_tasks": float(g["mean_latency_sec"].mean()),
            "mean_vram_gb_across_tasks": float(g["peak_vram_gb"].mean()),
            "mean_flops_per_token": float(g["flops_per_token"].mean()),
        })

    out = pd.DataFrame(rows).sort_values(
        ["weighted_accuracy", "mean_latency_sec_across_tasks"],
        ascending=[False, True]
    ).reset_index(drop=True)
    return out

def exact_two_sided_binom_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob_obs = math.comb(n, k) * (0.5 ** n)
    p = 0.0
    for i in range(0, n + 1):
        prob_i = math.comb(n, i) * (0.5 ** n)
        if prob_i <= prob_obs + 1e-15:
            p += prob_i
    return min(1.0, p)

def pairwise_stats(df):
    required = {"dataset_name", "strategy", "requested_model_name", "sample_id", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns for pairwise stats: {sorted(missing)}")

    rows = []
    for (dataset, strategy), g in df.groupby(["dataset_name", "strategy"], dropna=False):
        models = sorted(g["requested_model_name"].dropna().unique().tolist())
        for a, b in itertools.combinations(models, 2):
            ga = g[g["requested_model_name"] == a][["sample_id", "correct"]].rename(columns={"correct": "correct_a"})
            gb = g[g["requested_model_name"] == b][["sample_id", "correct"]].rename(columns={"correct": "correct_b"})
            m = ga.merge(gb, on="sample_id", how="inner")
            if m.empty:
                continue

            m["correct_a"] = pd.to_numeric(m["correct_a"], errors="coerce").fillna(0).astype(int)
            m["correct_b"] = pd.to_numeric(m["correct_b"], errors="coerce").fillna(0).astype(int)

            b01 = int(((m["correct_a"] == 0) & (m["correct_b"] == 1)).sum())
            b10 = int(((m["correct_a"] == 1) & (m["correct_b"] == 0)).sum())

            if (b01 + b10) == 0:
                stat = 0.0
                pval = 1.0
            else:
                stat = ((abs(b01 - b10) - 1.0) ** 2) / float(b01 + b10)
                pval = exact_two_sided_binom_pvalue(b01, b10)

            rows.append({
                "dataset": dataset,
                "strategy": strategy,
                "requested_model_a": a,
                "requested_model_b": b,
                "n_overlap": int(len(m)),
                "acc_a": float(m["correct_a"].mean()),
                "acc_b": float(m["correct_b"].mean()),
                "n01_a0_b1": b01,
                "n10_a1_b0": b10,
                "mcnemar_stat": float(stat),
                "p_value": float(pval),
            })

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "strategy", "requested_model_a", "requested_model_b"]
    ).reset_index(drop=True)
    return out

def build_error_pack(df):
    bad = df[df["correct"].astype(int) == 0].copy()
    keep_cols = [
        "requested_model_name",
        "actual_model_name",
        "dataset_name",
        "strategy",
        "sample_id",
        "question",
        "gold_answer",
        "prediction",
        "response_text",
    ]
    keep_cols = [c for c in keep_cols if c in bad.columns]
    bad = bad[keep_cols].copy()
    bad["manual_error_type"] = ""
    bad["manual_notes"] = ""
    return bad

def save_prompting_tables(agg):
    acc = agg.pivot_table(
        index=["requested_model_name", "actual_model_name"],
        columns=["dataset_name", "strategy"],
        values="accuracy"
    ).reset_index()
    lat = agg.pivot_table(
        index=["requested_model_name", "actual_model_name"],
        columns=["dataset_name", "strategy"],
        values="mean_latency_sec"
    ).reset_index()

    acc.columns = ["__".join([str(x) for x in col if str(x) != ""]) for col in acc.columns.to_flat_index()]
    lat.columns = ["__".join([str(x) for x in col if str(x) != ""]) for col in lat.columns.to_flat_index()]

    acc_path = AGG_DIR / "prompting_accuracy_final.csv"
    lat_path = AGG_DIR / "prompting_latency_final.csv"
    acc.to_csv(acc_path, index=False)
    lat.to_csv(lat_path, index=False)
    print(f"Saved: {acc_path}")
    print(f"Saved: {lat_path}")

def plot_pareto_by_strategy(weighted):
    for strategy in sorted(weighted["strategy"].dropna().unique()):
        d = weighted[weighted["strategy"] == strategy].copy()
        if d.empty:
            continue
        plt.figure(figsize=(7, 5))
        plt.scatter(d["mean_latency_sec_across_tasks"], d["weighted_accuracy"])
        for _, r in d.iterrows():
            plt.annotate(
                r["requested_model_name"],
                (r["mean_latency_sec_across_tasks"], r["weighted_accuracy"])
            )
        plt.xlabel("Mean latency across tasks (sec)")
        plt.ylabel("Weighted accuracy")
        plt.title(f"Pareto-style comparison: {strategy}")
        plt.tight_layout()
        out = FIG_DIR / f"pareto_{strategy}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved: {out}")

def plot_prompting_accuracy_per_dataset(agg):
    for dataset in sorted(agg["dataset_name"].dropna().unique()):
        d = agg[agg["dataset_name"] == dataset].copy()
        if d.empty:
            continue
        pivot = d.pivot(index="requested_model_name", columns="strategy", values="accuracy")
        pivot = pivot.sort_index()
        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Prompting accuracy on {dataset}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = FIG_DIR / f"prompting_accuracy_{dataset}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved: {out}")

def plot_scaling(weighted, agg):
    params = (
        agg.groupby(["requested_model_name", "actual_model_name"], dropna=False)["active_params_b"]
        .mean()
        .reset_index()
    )
    d = weighted.merge(params, on=["requested_model_name", "actual_model_name"], how="left")
    plt.figure(figsize=(7, 5))
    plt.scatter(d["active_params_b"], d["weighted_accuracy"])
    for _, r in d.iterrows():
        plt.annotate(
            f"{r['requested_model_name']} ({r['strategy']})",
            (r["active_params_b"], r["weighted_accuracy"])
        )
    plt.xlabel("Active parameters (B)")
    plt.ylabel("Weighted accuracy")
    plt.title("Scaling vs weighted accuracy")
    plt.tight_layout()
    out = FIG_DIR / "scaling_final.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")

def main():
    ensure_dirs()

    df = load_final_raw()
    raw_path = AGG_DIR / "raw_concat_final.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved: {raw_path}")

    agg = aggregate(df)
    agg_path = AGG_DIR / "aggregated_final.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    weighted = weighted_summary(agg)
    weighted_path = AGG_DIR / "weighted_summary_final.csv"
    weighted.to_csv(weighted_path, index=False)
    print(f"Saved: {weighted_path}")

    pairwise = pairwise_stats(df)
    pairwise_path = AGG_DIR / "pairwise_stats_final.csv"
    pairwise.to_csv(pairwise_path, index=False)
    print(f"Saved: {pairwise_path}")

    error_pack = build_error_pack(df)
    error_path = AGG_DIR / "error_pack_for_manual_review.csv"
    error_pack.to_csv(error_path, index=False)
    print(f"Saved: {error_path}")

    save_prompting_tables(agg)
    plot_pareto_by_strategy(weighted)
    plot_prompting_accuracy_per_dataset(agg)
    plot_scaling(weighted, agg)

if __name__ == "__main__":
    main()
