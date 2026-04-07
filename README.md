# dense_and_moe_reasoning

This repository contains the code, prompts, prepared datasets, analysis scripts, and released artifacts for a controlled benchmark of recent dense and mixture-of-experts (MoE) reasoning language models. The project compares model accuracy, latency, VRAM usage, and an approximate FLOPs-per-token proxy across multiple reasoning benchmarks and prompting strategies.

## What this repository does

The benchmark evaluates seven open-weight models across four datasets and three prompting strategies under a unified evaluation pipeline. The goal is to study not only which models are accurate, but which models offer the strongest practical accuracy--efficiency tradeoffs under realistic inference constraints.

### Models

- `gemma_4_e2b`
- `gemma_4_e4b`
- `gemma_4_26b_a4b`
- `phi_4_mini_reasoning`
- `phi_4_reasoning`
- `qwen3_8b`
- `qwen3_30b_a3b`

### Datasets

- `arc_challenge`
- `gsm8k`
- `math_l1_l3`
- `truthfulqa_mc1`

### Prompting strategies

- `zero_shot`
- `cot`
- `few_shot_cot`

Each model--dataset--strategy condition is evaluated on 100 examples, giving a total of 8,400 scored examples in the final released benchmark.

## Main result snapshot

The released benchmark shows that no single model dominates every task and prompting regime.

- Best overall weighted configuration: `gemma_4_e4b` with `few_shot_cot`, weighted accuracy **0.675**
- Second-best overall weighted configuration: `gemma_4_26b_a4b` with `few_shot_cot`, weighted accuracy **0.663**
- Strongest overall low-memory compromise: `gemma_4_e2b`
- Strongest TruthfulQA behavior: Phi models
- Strongest ARC and Math behavior: Gemma models
- Strongest prompt sensitivity: GSM8K

The broader conclusion is that sparse activation alone does not guarantee the best practical operating point. Observed tradeoffs depend jointly on architecture, prompting protocol, task family, and deployment constraints.

## Repository structure

```text
.
├── analysis/
├── archive_cleanup/
├── archive_model_swap/
├── configs/
├── data/
│   ├── indices/
│   └── prepared/
├── evaluation/
├── experiments/
├── figures/
├── logs/
├── models/
├── notebooks/
├── prompts/
├── results/
│   ├── aggregated/
│   └── raw/
├── reproduce.sh
├── run_complete_plan_v3.sh
├── run_coverage_v2.sh
└── README.md
```

## Important directories

### `configs/`

Configuration files for datasets, models, and prompting.

* `configs/datasets.yaml`
* `configs/models.yaml`
* `configs/prompts.yaml`

### `data/prepared/`

Prepared benchmark subsets used in the final evaluation.

* `arc_challenge.jsonl`
* `gsm8k.jsonl`
* `math_l1_l3.jsonl`
* `truthfulqa_mc1.jsonl`

### `prompts/`

Prompt construction code and few-shot examples.

* `prompts/builder.py`
* `prompts/few_shot_examples.json`

### `models/`

Model loading and inference helpers.

* `models/loader.py`
* `models/inference.py`

### `evaluation/`

Answer extraction, grading, and metric utilities.

* `evaluation/extractor.py`
* `evaluation/grader.py`
* `evaluation/metrics.py`

### `experiments/`

Main benchmark entrypoints.

* `experiments/run_benchmark_final_clean.py`
* `experiments/run_benchmark_with_fallback.py`
* `experiments/run_experiment.py`
* `experiments/run_single.py`

### `analysis/`

Aggregation, figure generation, and result finalization.

* `analysis/finalize_complete_plan_results.py`
* `analysis/final_v2_aggregate.py`
* `analysis/final_v2_pairwise.py`
* `analysis/final_v2_figures.py`
* `analysis/final_v2_error_pack.py`
* `analysis/repair_pairwise_stats_final.py`
* `analysis/print_results_for_paper.py`

### `results/raw/`

Per-run raw evaluation outputs.

### `results/aggregated/`

Final released CSV artifacts used in the paper.

### `figures/`

Final released figures used in the paper.

## Final released artifacts

### Aggregated CSV files

Located in `results/aggregated/`:

* `aggregated_final.csv`
* `weighted_summary_final.csv`
* `pairwise_stats_final.csv`
* `raw_concat_final.csv`
* `error_pack_for_manual_review.csv`
* `prompting_accuracy_final.csv`
* `prompting_latency_final.csv`

### Figures

Located in `figures/`:

* `pareto_zero_shot.png`
* `pareto_cot.png`
* `pareto_few_shot_cot.png`
* `prompting_accuracy_arc_challenge.png`
* `prompting_accuracy_gsm8k.png`
* `prompting_accuracy_math_l1_l3.png`
* `prompting_accuracy_truthfulqa_mc1.png`
* `scaling_final.png`

## Released benchmark sizes

The final released benchmark has the following expected sizes:

* Raw evaluated rows: **8400**
* Aggregated rows: **84**
* Weighted summary rows: **21**
* Pairwise rows: **252**

These counts are useful for validating a successful reproduction.

## Metrics

The benchmark reports:

* Accuracy
* 95% confidence intervals
* Latency
* Output length
* Tokens per second
* Peak VRAM usage
* Approximate FLOPs per token
* Weighted cross-task accuracy
* Pairwise McNemar-style significance comparisons

### Weighted summary task weights

* GSM8K = 0.40
* Math L1--L3 = 0.30
* ARC-Challenge = 0.20
* TruthfulQA MC1 = 0.10

## Environment used for the released run

The artifact inventory for the benchmark server reports:

* Host: `axis2`
* OS: Ubuntu 22.04.5 LTS
* Python: 3.13.11
* PyTorch: 2.6.0+cu124
* Transformers: 5.5.0
* Accelerate: 1.13.0
* Pandas: 3.0.1
* NumPy: 2.4.3
* SciPy: 1.17.1
* Matplotlib: 3.10.8
* CUDA driver version: 550.144.03
* CUDA version: 12.4
* GPU platform: NVIDIA H100 80GB HBM3

## How to run

This section gives a practical workflow using the scripts already present in the repository. Depending on your local setup, model availability, and credentials, you may run either the full benchmark scripts or smaller single-condition experiments.

### 1. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

If you already use a project environment, activate that instead.

### 2. Prepare datasets

If the prepared files are already present in `data/prepared/`, you can skip this step. Otherwise run:

```bash
python data/prepare_datasets.py
```

### 3. Run a single experiment

For a smaller smoke test or debugging run, use one of the experiment entrypoints:

```bash
python experiments/run_single.py
```

or

```bash
python experiments/run_experiment.py
```

If your local version of these scripts expects arguments, inspect the script header or run with `-h` first.

### 4. Run the benchmark pipeline

The repository includes benchmark entrypoints and shell wrappers. The most relevant ones are:

```bash
python experiments/run_benchmark_final_clean.py
```

or

```bash
python experiments/run_benchmark_with_fallback.py
```

and the shell wrappers:

```bash
bash reproduce.sh
```

```bash
bash run_complete_plan_v3.sh
```

```bash
bash run_coverage_v2.sh
```

### 5. Aggregate the raw outputs

Once raw CSVs have been generated under `results/raw/`, finalize the released outputs with:

```bash
python analysis/finalize_complete_plan_results.py
```

If needed, repair the pairwise results with:

```bash
python analysis/repair_pairwise_stats_final.py
```

### 6. Regenerate the figures

```bash
python analysis/final_v2_figures.py
```

### 7. Check the final outputs

After a successful run, the main released files should appear under:

```text
results/aggregated/
figures/
```

In particular, verify the presence of:

```text
results/aggregated/aggregated_final.csv
results/aggregated/weighted_summary_final.csv
results/aggregated/pairwise_stats_final.csv
results/aggregated/raw_concat_final.csv
results/aggregated/error_pack_for_manual_review.csv
results/aggregated/prompting_accuracy_final.csv
results/aggregated/prompting_latency_final.csv
```

and

```text
figures/pareto_zero_shot.png
figures/pareto_cot.png
figures/pareto_few_shot_cot.png
figures/prompting_accuracy_arc_challenge.png
figures/prompting_accuracy_gsm8k.png
figures/prompting_accuracy_math_l1_l3.png
figures/prompting_accuracy_truthfulqa_mc1.png
figures/scaling_final.png
```

## Minimal reproduction workflow

If you want the shortest practical sequence, use this order:

```bash
python -m venv .venv
source .venv/bin/activate
python data/prepare_datasets.py
bash run_complete_plan_v3.sh
python analysis/finalize_complete_plan_results.py
python analysis/repair_pairwise_stats_final.py
python analysis/final_v2_figures.py
```

## Notes on running

* The benchmark was developed and released in a server environment with H100 GPUs.
* Some models may require substantial VRAM.
* The framework contains fallback logic for model loading, but the final released benchmark is based on the intended evaluated model set.
* Archival directories such as `archive_cleanup/` and `archive_model_swap/` preserve intermediate experiment states and should not be confused with the final released artifacts.
* The authoritative final outputs for the paper are in `results/aggregated/` and `figures/`.

## Error analysis artifact

The file

```text
results/aggregated/error_pack_for_manual_review.csv
```

contains the incorrect raw rows from the released benchmark and is intended to support qualitative inspection of failure modes, extraction issues, and prompt-sensitive errors.

## Suggested validation checks

After running the full pipeline, verify:

* `results/aggregated/aggregated_final.csv` has **84** rows
* `results/aggregated/weighted_summary_final.csv` has **21** rows
* `results/aggregated/pairwise_stats_final.csv` has **252** rows
* the final figures are present in `figures/`

## Repository history note

This repository contains both final released artifacts and archival experiment directories. For reproducing the reported benchmark, use the current code under the main top-level directories and treat the following as archival:

* `archive_cleanup/`
* `archive_model_swap/`

## Citation

If you use this repository or its artifacts, please cite the corresponding paper once finalized.

## Repository URL

[https://github.com/mkboch/dense_and_moe_reasoning](https://github.com/mkboch/dense_and_moe_reasoning)
```
