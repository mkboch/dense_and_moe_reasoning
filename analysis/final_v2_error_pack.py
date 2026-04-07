from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated"
TAG = "coverage_v2"

def main():
    files = sorted(RAW_DIR.glob(f"{TAG}__*__*__*__*__n*.csv"))
    files = [f for f in files if "__ALL_RUNS__" not in f.name]
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    target = raw[(raw["dataset_name"].isin({"gsm8k","math_l1_l3"})) & (raw["correct"] == 0)].copy()
    pack = []
    for (requested_model_name, dataset_name), g in target.groupby(["requested_model_name","dataset_name"], dropna=False):
        t = g.head(25).copy()
        t["manual_error_type"] = ""
        t["manual_notes"] = ""
        pack.append(t)

    out = pd.concat(pack, ignore_index=True)
    keep = [
        "requested_model_name","actual_model_name","dataset_name","strategy","sample_id",
        "question","gold_answer","prediction","response_text","manual_error_type","manual_notes"
    ]
    out = out[keep]
    out.to_csv(AGG_DIR / "coverage_v2_error_pack_for_manual_review.csv", index=False)
    print(f"Saved: {AGG_DIR / 'coverage_v2_error_pack_for_manual_review.csv'}")
    print(out.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
