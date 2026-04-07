import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROMPT_CFG_PATH = ROOT / "configs" / "prompts.yaml"
FEW_SHOT_PATH = ROOT / "prompts" / "few_shot_examples.json"


def load_prompt_config() -> Dict[str, Any]:
    with open(PROMPT_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_few_shot_examples() -> Dict[str, List[Dict[str, str]]]:
    with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_few_shot_block(dataset_name: str) -> str:
    examples = load_few_shot_examples().get(dataset_name, [])
    blocks = []
    for ex in examples:
        block = (
            f"Q: {ex['question']}\n"
            f"A: {ex['reasoning']}\n"
            f"#### {ex['answer']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_prompt(strategy: str, question: str, dataset_name: str) -> str:
    cfg = load_prompt_config()
    strategies = cfg["strategies"]

    if strategy not in strategies:
        raise ValueError(f"Unknown prompting strategy: {strategy}")

    template = strategies[strategy]["template"]

    if strategy == "few_shot_cot":
        prompt = template.format(
            few_shot_block=build_few_shot_block(dataset_name),
            question=question,
        )
    else:
        prompt = template.format(question=question)

    if dataset_name in {"gsm8k", "math_l1_l3"}:
        prompt += (
            "\n\nRules:\n"
            "1. Reason in no more than 4 short steps.\n"
            "2. The final line must be exactly: #### <answer>\n"
            "3. Do not output anything after that final line."
        )
    elif dataset_name in {"arc_challenge", "truthfulqa_mc1"}:
        prompt += (
            "\n\nRules:\n"
            "1. Return only one capital letter.\n"
            "2. Valid answers are A, B, C, D, or E.\n"
            "3. Do not output any explanation."
        )

    return prompt


def maybe_apply_chat_template(tokenizer: Any, prompt: str, use_chat_template: bool = True) -> str:
    if not use_chat_template:
        return prompt

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered
    except Exception:
        return prompt


def get_generation_config() -> Dict[str, Any]:
    cfg = load_prompt_config()
    return cfg["generation"]
