import re
from typing import Optional


def _clean_number(text: str) -> str:
    text = text.replace(",", "").strip()
    text = text.strip("$")
    return text.strip()


def extract_gsm8k(response: str) -> Optional[str]:
    m = re.search(r"####\s*\$?\s*([-+]?\d+(?:\.\d+)?)\s*$", response.strip(), flags=re.MULTILINE)
    if m:
        return _clean_number(m.group(1))

    m = re.search(r"answer\s+is\s+\$?\s*([-+]?\d+(?:\.\d+)?)\s*$", response.strip(), flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return _clean_number(m.group(1))

    lines = [x.strip() for x in response.splitlines() if x.strip()]
    if lines:
        last = lines[-1]
        m = re.fullmatch(r"\$?\s*([-+]?\d+(?:\.\d+)?)", last)
        if m:
            return _clean_number(m.group(1))

    return None


def extract_math(response: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", response)
    if boxed:
        return boxed[-1].strip()

    m = re.search(r"####\s*(.+?)\s*$", response.strip(), flags=re.MULTILINE)
    if m:
        return m.group(1).strip().strip("$")

    lines = [x.strip() for x in response.splitlines() if x.strip()]
    if lines:
        last = lines[-1]
        if last:
            return last.strip().strip("$")

    return None


def extract_arc(response: str) -> Optional[str]:
    text = response.strip().upper()

    if re.fullmatch(r"[A-E]", text):
        return text

    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1)

    return None


def extract_truthfulqa(response: str) -> Optional[str]:
    return extract_arc(response)


def extract_answer(response: str, extractor_name: str) -> Optional[str]:
    if extractor_name == "gsm8k":
        return extract_gsm8k(response)
    if extractor_name == "math":
        return extract_math(response)
    if extractor_name == "arc":
        return extract_arc(response)
    if extractor_name == "truthfulqa":
        return extract_truthfulqa(response)
    raise ValueError(f"Unknown extractor: {extractor_name}")
