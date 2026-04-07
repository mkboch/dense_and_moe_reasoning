import re
from typing import Optional


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def normalize_number_like(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip().replace(",", "")
    t = re.sub(r"\s+", "", t)
    return t.lower()


def extract_gold_gsm8k(gold_answer: str) -> Optional[str]:
    match = re.search(r"####\s*([\-]?\d[\d,\.]*)", gold_answer)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def extract_gold_math(gold_answer: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", gold_answer)
    if boxed:
        return boxed[-1].strip()
    return gold_answer.strip()


def grade_prediction(prediction: Optional[str], gold_answer: str, dataset_name: str) -> int:
    if prediction is None:
        return 0

    if dataset_name == "gsm8k":
        gold = extract_gold_gsm8k(gold_answer)
        return int(normalize_number_like(prediction) == normalize_number_like(gold))

    if dataset_name == "math_l1_l3":
        gold = extract_gold_math(gold_answer)
        return int(normalize_number_like(prediction) == normalize_number_like(gold))

    if dataset_name in {"arc_challenge", "truthfulqa_mc1"}:
        return int(normalize_text(prediction) == normalize_text(gold_answer))

    raise ValueError(f"Unknown dataset for grading: {dataset_name}")
