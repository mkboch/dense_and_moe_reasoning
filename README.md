# Dense and MoE Reasoning LLMs: Benchmarking Gemma 4, Phi-4, and Qwen3

This repository contains the code, prompts, prepared datasets, analysis scripts, and released artifacts for a controlled benchmark of recent dense and mixture-of-experts (MoE) reasoning-oriented language models. The study compares accuracy, latency, VRAM usage, and an approximate FLOPs-per-token proxy across multiple reasoning tasks and prompting strategies.

## Overview

The benchmark studies seven open-weight models:

- `gemma_4_e2b`
- `gemma_4_e4b`
- `gemma_4_26b_a4b`
- `phi_4_mini_reasoning`
- `phi_4_reasoning`
- `qwen3_8b`
- `qwen3_30b_a3b`

across four datasets:

- `arc_challenge`
- `gsm8k`
- `math_l1_l3`
- `truthfulqa_mc1`

and three prompting strategies:

- `zero_shot`
- `cot`
- `few_shot_cot`

Each model--dataset--strategy condition is evaluated on 100 examples, yielding a total of 8,400 scored examples.

The goal of the project is not only to compare raw accuracy, but to study prompt-conditioned accuracy--efficiency tradeoffs under a unified evaluation pipeline.

## Main findings

The main empirical pattern is that no single model dominates all tasks and prompting regimes.

- **Best overall weighted result:** `gemma_4_e4b` under `few_shot_cot`, with weighted accuracy **0.675**
- **Second-best overall weighted result:** `gemma_4_26b_a4b` under `few_shot_cot`, with weighted accuracy **0.663**
- **Best low-memory all-round option:** `gemma_4_e2b`
- **Strongest TruthfulQA behavior:** Phi models, especially `phi_4_reasoning`
- **Strongest ARC and Math behavior:** Gemma models
- **Largest prompting sensitivity:** GSM8K, especially for `phi_4_reasoning`

A key conclusion of the benchmark is that sparse activation alone does not guarantee the best practical operating point. Observed tradeoffs depend jointly on architecture, prompt protocol, task family, and deployment constraints.

## Repository layout

```text
.
├── analysis/
├── configs/
├── data/
│   ├── indices/
│   └── prepared/
├── evaluation/
├── experiments/
├── figures/
├── models/
├── prompts/
├── results/
│   ├── aggregated/
│   └── raw/
├── notebooks/
├── reproduce.sh
├── run_complete_plan_v3.sh
├── run_coverage_v2.sh
└── requirements.txt
