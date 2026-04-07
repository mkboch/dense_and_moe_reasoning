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
├── requirements.txt
└── README.md
