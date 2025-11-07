import re
from typing import Iterable

PROMPT_TEMPLATE = """
You are given a multimodal reasoning problem.

Question:
{question}

Caption:
{caption}

Answer the question twice:
1. Provide a direct answer without referencing the caption.
2. Provide an answer that can rely on the caption if it helps.

Return the responses strictly in this XML format:
<direct_answer>...</direct_answer>
<caption_answer>...</caption_answer>
""".strip()

_CAPTION_EXTRACT = re.compile(r"<caption>(.*?)</caption>", re.DOTALL | re.IGNORECASE)
_ANSWER_PATTERN = {
    "direct": re.compile(r"<direct_answer>(.*?)</direct_answer>", re.DOTALL | re.IGNORECASE),
    "caption": re.compile(r"<caption_answer>(.*?)</caption_answer>", re.DOTALL | re.IGNORECASE),
}


def _extract_caption(text: str) -> str:
    if not text:
        return ""
    match = _CAPTION_EXTRACT.search(text)
    return match.group(1).strip() if match else ""


def construct_gain_inputs_from_rollouts(rollout_question: str, rollout_response: str, ground_truth=None) -> str:
    caption = _extract_caption(rollout_response)
    formatted_caption = caption if caption else "(no caption provided)"
    question = rollout_question.strip() if rollout_question else "(question unavailable)"
    return PROMPT_TEMPLATE.format(question=question, caption=formatted_caption)


def _normalize(ans: str | None) -> str:
    if not ans:
        return ""
    return re.sub(r"\s+", " ", ans.strip()).lower()


def _extract_answers(text: str) -> tuple[str, str]:
    direct = _ANSWER_PATTERN["direct"].search(text or "")
    caption = _ANSWER_PATTERN["caption"].search(text or "")
    return _normalize(direct.group(1) if direct else None), _normalize(caption.group(1) if caption else None)


def convert_gain_responses_to_rewards(text: str | Iterable[str], meta: dict | None = None) -> float:
    direct_ans, caption_ans = _extract_answers(text if isinstance(text, str) else "\n".join(text))
    ground_truth = _normalize((meta or {}).get("ground_truth"))
    direct_correct = ground_truth and direct_ans == ground_truth
    caption_correct = ground_truth and caption_ans == ground_truth

    if caption_correct and not direct_correct:
        return 1.0
    if caption_correct and direct_correct:
        return 0.7
    if not caption_correct and not direct_correct:
        return 0.2
    return 0.0
