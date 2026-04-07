from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "aggregated" / "aggregated_final.csv"
WEI = ROOT / "results" / "aggregated" / "weighted_summary_final.csv"
PAIR = ROOT / "results" / "aggregated" / "pairwise_stats_final.csv"
RAW = ROOT / "results" / "aggregated" / "raw_concat_final.csv"

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

agg = pd.read_csv(AGG)
wei = pd.read_csv(WEI)
pair = pd.read_csv(PAIR)
raw = pd.read_csv(RAW)

print("\n====================")
print("SANITY CHECK")
print("====================")
print(f"aggregated rows: {len(agg)}")
print(f"weighted rows:   {len(wei)}")
print(f"pairwise rows:   {len(pair)}")
print(f"raw rows:        {len(raw)}")
print("\nModels in aggregated:")
print(sorted(agg['requested_model_name'].unique().tolist()))
print("\nDatasets:")
print(sorted(agg['dataset_name'].unique().tolist()))
print("\nStrategies:")
print(sorted(agg['strategy'].unique().tolist()))

print("\n====================")
print("FINAL WEIGHTED SUMMARY (sorted by weighted_accuracy)")
print("====================")
print(
    wei.sort_values(["weighted_accuracy", "mean_latency_sec_across_tasks"], ascending=[False, True])[
        [
            "requested_model_name",
            "strategy",
            "weighted_accuracy",
            "mean_latency_sec_across_tasks",
            "mean_vram_gb_across_tasks",
            "mean_flops_per_token",
        ]
    ].to_string(index=False)
)

print("\n====================")
print("BEST STRATEGY PER MODEL")
print("====================")
best_strategy_per_model = (
    wei.sort_values(["requested_model_name", "weighted_accuracy", "mean_latency_sec_across_tasks"], ascending=[True, False, True])
       .groupby("requested_model_name", as_index=False)
       .first()
)
print(
    best_strategy_per_model[
        [
            "requested_model_name",
            "strategy",
            "weighted_accuracy",
            "mean_latency_sec_across_tasks",
            "mean_vram_gb_across_tasks",
        ]
    ].to_string(index=False)
)

print("\n====================")
print("BEST MODEL PER STRATEGY")
print("====================")
best_model_per_strategy = (
    wei.sort_values(["strategy", "weighted_accuracy", "mean_latency_sec_across_tasks"], ascending=[True, False, True])
       .groupby("strategy", as_index=False)
       .first()
)
print(
    best_model_per_strategy[
        [
            "strategy",
            "requested_model_name",
            "weighted_accuracy",
            "mean_latency_sec_across_tasks",
            "mean_vram_gb_across_tasks",
        ]
    ].to_string(index=False)
)

for dataset in sorted(agg["dataset_name"].unique()):
    print("\n====================")
    print(f"DATASET: {dataset}")
    print("====================")
    sub = agg[agg["dataset_name"] == dataset].copy()

    print("\nTop models within each strategy:")
    for strategy in sorted(sub["strategy"].unique()):
        ss = sub[sub["strategy"] == strategy].sort_values(
            ["accuracy", "mean_latency_sec", "peak_vram_gb"],
            ascending=[False, True, True]
        )
        print(f"\n--- {strategy} ---")
        print(
            ss[
                [
                    "requested_model_name",
                    "accuracy",
                    "ci_low",
                    "ci_high",
                    "mean_latency_sec",
                    "mean_output_tokens",
                    "peak_vram_gb",
                ]
            ].to_string(index=False)
        )

    print("\nBest row for this dataset overall:")
    best_row = sub.sort_values(["accuracy", "mean_latency_sec"], ascending=[False, True]).iloc[0]
    second_row = sub.sort_values(["accuracy", "mean_latency_sec"], ascending=[False, True]).iloc[1]
    print(pd.DataFrame([best_row, second_row])[
        [
            "requested_model_name",
            "strategy",
            "accuracy",
            "ci_low",
            "ci_high",
            "mean_latency_sec",
            "peak_vram_gb",
        ]
    ].to_string(index=False))
    print(f"\nGap between best and second-best accuracy on {dataset}: {best_row['accuracy'] - second_row['accuracy']:.4f}")

print("\n====================")
print("BEST STRATEGY FOR EACH MODEL ON EACH DATASET")
print("====================")
best_per_model_dataset = (
    agg.sort_values(
        ["requested_model_name", "dataset_name", "accuracy", "mean_latency_sec"],
        ascending=[True, True, False, True]
    )
    .groupby(["requested_model_name", "dataset_name"], as_index=False)
    .first()
)
print(
    best_per_model_dataset[
        [
            "requested_model_name",
            "dataset_name",
            "strategy",
            "accuracy",
            "mean_latency_sec",
            "peak_vram_gb",
        ]
    ].to_string(index=False)
)

print("\n====================")
print("PROMPTING EFFECT WITHIN EACH MODEL/DATASET")
print("====================")
prompt_effect_rows = []
for (model, dataset), g in agg.groupby(["requested_model_name", "dataset_name"]):
    g2 = g.sort_values("accuracy", ascending=False).reset_index(drop=True)
    best = g2.iloc[0]
    worst = g2.iloc[-1]
    prompt_effect_rows.append({
        "requested_model_name": model,
        "dataset_name": dataset,
        "best_strategy": best["strategy"],
        "best_accuracy": best["accuracy"],
        "worst_strategy": worst["strategy"],
        "worst_accuracy": worst["accuracy"],
        "spread": best["accuracy"] - worst["accuracy"],
    })
prompt_effect = pd.DataFrame(prompt_effect_rows).sort_values(["spread", "requested_model_name", "dataset_name"], ascending=[False, True, True])
print(prompt_effect.to_string(index=False))

print("\n====================")
print("WIN COUNTS ACROSS THE 12 DATASET-STRATEGY CONDITIONS")
print("====================")
win_counts = (
    agg.sort_values(["dataset_name", "strategy", "accuracy", "mean_latency_sec"], ascending=[True, True, False, True])
       .groupby(["dataset_name", "strategy"], as_index=False)
       .first()
       .groupby("requested_model_name")
       .size()
       .reset_index(name="num_condition_wins")
       .sort_values(["num_condition_wins", "requested_model_name"], ascending=[False, True])
)
print(win_counts.to_string(index=False))

print("\n====================")
print("LOWEST-LATENCY ROW WITH ACCURACY >= 0.80")
print("====================")
fast_high_acc = agg[agg["accuracy"] >= 0.80].sort_values(["mean_latency_sec", "accuracy"], ascending=[True, False])
if len(fast_high_acc) > 0:
    print(
        fast_high_acc[
            [
                "requested_model_name",
                "dataset_name",
                "strategy",
                "accuracy",
                "mean_latency_sec",
                "peak_vram_gb",
            ]
        ].to_string(index=False)
    )
else:
    print("No rows found with accuracy >= 0.80")

print("\n====================")
print("PARETO-LIKE VIEW FROM WEIGHTED SUMMARY")
print("Sorted by weighted_accuracy desc, latency asc, VRAM asc")
print("====================")
pareto_view = wei.sort_values(
    ["weighted_accuracy", "mean_latency_sec_across_tasks", "mean_vram_gb_across_tasks"],
    ascending=[False, True, True]
)
print(
    pareto_view[
        [
            "requested_model_name",
            "strategy",
            "weighted_accuracy",
            "mean_latency_sec_across_tasks",
            "mean_vram_gb_across_tasks",
            "mean_flops_per_token",
        ]
    ].to_string(index=False)
)

print("\n====================")
print("SIGNIFICANT PAIRWISE RESULTS (p < 0.05)")
print("====================")
sig = pair[pair["p_value"] < 0.05].copy()
sig = sig.sort_values(["dataset", "strategy", "p_value", "mcnemar_stat"], ascending=[True, True, True, False])
print(f"Number of significant pairwise rows: {len(sig)} / {len(pair)}")
print(
    sig[
        [
            "dataset",
            "strategy",
            "requested_model_a",
            "requested_model_b",
            "acc_a",
            "acc_b",
            "n01_a0_b1",
            "n10_a1_b0",
            "mcnemar_stat",
            "p_value",
        ]
    ].to_string(index=False)
)

print("\n====================")
print("MOST IMPORTANT PAIRWISE COMPARISONS FOR RESULTS WRITING")
print("====================")
priority_pairs = [
    ("gemma_4_e4b", "gemma_4_e2b"),
    ("gemma_4_e4b", "gemma_4_26b_a4b"),
    ("gemma_4_e4b", "qwen3_8b"),
    ("gemma_4_e4b", "qwen3_30b_a3b"),
    ("gemma_4_e4b", "phi_4_reasoning"),
    ("gemma_4_e4b", "phi_4_mini_reasoning"),
]
priority = []
for a, b in priority_pairs:
    tmp = pair[((pair["requested_model_a"] == a) & (pair["requested_model_b"] == b)) |
               ((pair["requested_model_a"] == b) & (pair["requested_model_b"] == a))]
    priority.append(tmp)
priority = pd.concat(priority, ignore_index=True).drop_duplicates()
priority = priority.sort_values(["dataset", "strategy", "p_value"], ascending=[True, True, True])
print(
    priority[
        [
            "dataset",
            "strategy",
            "requested_model_a",
            "requested_model_b",
            "acc_a",
            "acc_b",
            "mcnemar_stat",
            "p_value",
        ]
    ].to_string(index=False)
)

print("\n====================")
print("ERROR PACK SIZE")
print("====================")
err = raw[raw["correct"] == 0].copy()
print(f"Total incorrect raw rows: {len(err)}")
print("\nIncorrect counts by dataset/strategy/model:")
print(
    err.groupby(["dataset_name", "strategy", "requested_model_name"])
       .size()
       .reset_index(name="num_errors")
       .sort_values(["dataset_name", "strategy", "num_errors"], ascending=[True, True, False])
       .to_string(index=False)
)
