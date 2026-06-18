from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Callable


LMEV2EvalJudge = Callable[[str, str, str], bool]


@dataclass(frozen=True, slots=True)
class LongMemEvalV2Score:
    eval_name: str
    eval_function: str
    prediction: str
    reference: str
    parsed_prediction: str
    score: bool | None
    skipped: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_eval_function_spec(spec: str) -> tuple[str, dict[str, object]]:
    parts = [part for part in str(spec or "").split("|") if part]
    name = parts[0] if parts else "norm_phrase_set_match"
    kwargs: dict[str, object] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        lowered = value.lower()
        if lowered == "true":
            kwargs[key] = True
        elif lowered == "false":
            kwargs[key] = False
        else:
            kwargs[key] = value
    return name, kwargs


def eval_name(spec: str) -> str:
    return parse_eval_function_spec(spec)[0]


def extract_boxed_answer(text: str) -> str:
    marker = r"\boxed{"
    start = str(text).rfind(marker)
    if start < 0:
        return str(text).strip()
    cursor = start + len(marker)
    depth = 1
    chars: list[str] = []
    while cursor < len(text) and depth:
        char = text[cursor]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth:
                chars.append(char)
        else:
            chars.append(char)
        cursor += 1
    parsed = "".join(chars).strip()
    return parsed or str(text).strip()


def normalize_phrase(text: str, *, lower: bool = True, normalize_hyphen: bool = True, strip_punct: bool = True) -> str:
    value = str(text)
    if lower:
        value = value.lower()
    if normalize_hyphen:
        value = value.replace("-", " ").replace("_", " ")
    value = re.sub(r"[,;]", " ", value)
    if strip_punct:
        value = re.sub(r"[^\w\s]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def split_phrases(text: str, *, separators: str | tuple[str, ...] = (",", ";"), **normalize_kwargs: object) -> list[str]:
    if isinstance(separators, str):
        separators_value = tuple(separators) if separators else (",", ";")
    else:
        separators_value = separators
    pattern = "|".join(re.escape(separator) for separator in separators_value)
    parts = re.split(pattern, str(text)) if pattern else [str(text)]
    return [phrase for phrase in (normalize_phrase(part, **_normalize_kwargs(normalize_kwargs)) for part in parts) if phrase]


def score_longmemeval_v2_answer(
    *,
    prediction: str,
    reference: str,
    eval_function: str,
    llm_judge: LMEV2EvalJudge | None = None,
) -> LongMemEvalV2Score:
    name, kwargs = parse_eval_function_spec(eval_function)
    parsed_prediction = extract_boxed_answer(prediction)
    try:
        if name == "norm_phrase_set_match":
            score = norm_phrase_set_match(prediction, reference, **kwargs)
        elif name == "norm_phrase_set_match_ordered":
            score = norm_phrase_set_match_ordered(prediction, reference, **kwargs)
        elif name == "mc_choice_match":
            score = mc_choice_match(prediction, reference, **kwargs)
        elif name == "mc_choice_set_match":
            score = mc_choice_set_match(prediction, reference, **kwargs)
        elif name in {"llm_abstention_checker", "llm_gotchas_checker"}:
            if llm_judge is None:
                return LongMemEvalV2Score(name, eval_function, prediction, reference, parsed_prediction, None, True, "llm_judge_disabled")
            score = bool(llm_judge(prediction, reference, name))
        else:
            return LongMemEvalV2Score(name, eval_function, prediction, reference, parsed_prediction, None, True, f"unsupported_eval_function:{name}")
    except Exception as exc:
        return LongMemEvalV2Score(name, eval_function, prediction, reference, parsed_prediction, None, True, f"scorer_error:{type(exc).__name__}:{exc}")
    return LongMemEvalV2Score(name, eval_function, prediction, reference, parsed_prediction, bool(score))


def norm_phrase_set_match(prediction: str, answer: str, **kwargs: object) -> bool:
    require_non_empty = bool(kwargs.pop("require_non_empty", False))
    separators = kwargs.pop("separators", (",", ";"))
    normalized_pred = normalize_phrase(prediction, **_normalize_kwargs(kwargs))
    phrases = sorted(set(split_phrases(answer, separators=separators, **kwargs)), key=len, reverse=True)
    if require_non_empty and not phrases:
        return False
    return all(re.search(rf"\b{re.escape(phrase)}\b", normalized_pred) for phrase in phrases)


def norm_phrase_set_match_ordered(prediction: str, answer: str, **kwargs: object) -> bool:
    require_non_empty = bool(kwargs.pop("require_non_empty", False))
    separators = kwargs.pop("separators", (",", ";"))
    normalized_pred = normalize_phrase(prediction, **_normalize_kwargs(kwargs))
    phrases = split_phrases(answer, separators=separators, **kwargs)
    if require_non_empty and not phrases:
        return False
    cursor = 0
    for phrase in phrases:
        match = re.search(rf"\b{re.escape(phrase)}\b", normalized_pred[cursor:])
        if not match:
            return False
        cursor += match.end()
    return True


def mc_choice_match(prediction: str, answer: str, **kwargs: object) -> bool:
    require_non_empty = bool(kwargs.get("require_non_empty", False))
    candidate = _normalize_choice(extract_boxed_answer(prediction))
    expected = _normalize_choice(answer)
    if require_non_empty and not candidate:
        return False
    return candidate == expected


def mc_choice_set_match(prediction: str, answer: str, **kwargs: object) -> bool:
    require_non_empty = bool(kwargs.get("require_non_empty", False))
    candidate = _extract_multi_select_letters(extract_boxed_answer(prediction))
    expected = _extract_multi_select_letters(answer)
    if require_non_empty and not candidate:
        return False
    return candidate == expected


def _normalize_kwargs(kwargs: dict[str, object]) -> dict[str, bool]:
    return {
        "lower": bool(kwargs.get("lower", True)),
        "normalize_hyphen": bool(kwargs.get("normalize_hyphen", True)),
        "strip_punct": bool(kwargs.get("strip_punct", True)),
    }


def _normalize_choice(text: str) -> str:
    value = str(text).strip().lower()
    boxed = re.search(r"\\boxed\{([^}]*)\}", value)
    if boxed:
        value = boxed.group(1)
    value = re.sub(r"\b(choice|option)\b", "", value, flags=re.IGNORECASE)
    return value.strip(f" \t\r\n{string.punctuation}").upper()


def _extract_multi_select_letters(text: str) -> frozenset[str]:
    value = _normalize_choice(text)
    value = re.sub(
        r"\b(AND|ANSWER|ANSWERS|CHOICE|CHOICES|FINAL|LETTER|LETTERS|OPTION|OPTIONS|SELECT|SELECTED|THE)\b",
        " ",
        value,
    )
    letters = re.findall(r"\b[A-Z]\b", value)
    if not letters and re.fullmatch(r"[A-Z]+", value):
        letters = list(value)
    return frozenset(letters)
