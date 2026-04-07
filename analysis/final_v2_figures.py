from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = ROOT / "results" / "aggregated"
FIG_DIR = ROOT / "figures"

def frontier(df):
    pts = df.sort_values(["flops_per_token","accuracy"], ascending=[True, False]).copy()
    keep = []
    best = -1
    for _, r in pts.iterrows():
        if r["accuracy"] > best:
            keep.append(True)
            best = r["accuracy"]
        else:
            keep.append(False)
    pts["is_frontier"] = keep
    return pts

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(AGG_DIR / "coverage_v2_aggregated.csv")

    for strategy in sorted(df["strategy"].unique()):
        sub = df[df["strategy"] == strategy].copy()
        pts = frontier(sub)
        plt.figure(figsize=(8,6))
        for _, r in pts.iterrows():
            plt.scatter(r["flops_per_token"], r["accuracy"])
            plt.text(r["flops_per_token"], r["accuracy"], f"{r['requested_model_name']}\n{r['dataset_name']}", fontsize=7)
        fr = pts[pts["is_frontier"]].sort_values("flops_per_token")
        if len(fr) >= 2:
            plt.plot(fr["flops_per_token"], fr["accuracy"])
        plt.xscale("log")
        plt.xlabel("FLOPs per token")
        plt.ylabel("Accuracy")
        plt.title(f"Pareto Frontier ({strategy})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"coverage_v2_pareto_{strategy}.png", dpi=200)
        plt.close()

    for dataset in sorted(df["dataset_name"].unique()):
        piv = df[df["dataset_name"] == dataset].pivot_table(index="requested_model_name", columns="strategy", values="accuracy", aggfunc="mean")
        plt.figure(figsize=(8,6))
        for col in piv.columns:
            plt.plot(piv.index, piv[col], marker="o", label=col)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title(f"Prompting Accuracy ({dataset})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"coverage_v2_prompting_{dataset}.png", dpi=200)
        plt.close()

    grp = df.groupby(["requested_model_name","architecture","active_params_b"], as_index=False)["accuracy"].mean()
    plt.figure(figsize=(8,6))
    for arch, g in grp.groupby("architecture"):
        x = g["active_params_b"].astype(float).values
        y = g["accuracy"].astype(float).values
        plt.scatter(x, y, label=arch)
        if len(x) >= 2:
            coeff = np.polyfit(np.log10(x), y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = coeff[0] * np.log10(xs) + coeff[1]
            plt.plot(xs, ys)
    plt.xscale("log")
    plt.xlabel("Active parameters (B)")
    plt.ylabel("Mean accuracy")
    plt.title("Scaling Accuracy vs Active Parameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "coverage_v2_scaling.png", dpi=200)
    plt.close()

    print("Saved figure set in figures/")

if __name__ == "__main__":
    main()
