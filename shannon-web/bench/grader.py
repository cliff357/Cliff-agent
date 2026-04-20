"""Task graders: exact match, regex, and LLM-as-judge."""
from __future__ import annotations
import json
import re
from typing import Any, Optional


def grade_exact(answer: str, expected_any: list[str]) -> dict:
    """Pass if answer (case-insensitive, substring) contains any expected phrase."""
    ans = (answer or "").strip().lower()
    for exp in expected_any:
        if exp.lower() in ans:
            return {"passed": True, "score": 1.0, "reason": f"Matched '{exp}'"}
    return {"passed": False, "score": 0.0, "reason": f"No match for any of {expected_any}"}


def grade_regex(answer: str, pattern: str) -> dict:
    """Pass if regex matches."""
    try:
        if re.search(pattern, answer or "", re.IGNORECASE | re.MULTILINE):
            return {"passed": True, "score": 1.0, "reason": f"Matched /{pattern}/"}
    except re.error as e:
        return {"passed": False, "score": 0.0, "reason": f"Invalid regex: {e}"}
    return {"passed": False, "score": 0.0, "reason": f"Pattern /{pattern}/ did not match"}


def grade_llm_judge(
    answer: str,
    rubric: str,
    question: str,
    judge_fn,
) -> dict:
    """Use an LLM judge to score 0-10.

    judge_fn: callable that takes a prompt and returns a string.
    Expected judge response is JSON: {"score": int 0-10, "reasoning": "..."}
    """
    judge_prompt = (
        "You are a strict, objective grader. You will score an answer from 0 to 10 "
        "based on the rubric.\n\n"
        f"## Question\n{question}\n\n"
        f"## Answer\n{answer}\n\n"
        f"## Rubric\n{rubric}\n\n"
        "Respond with a JSON object only, no other text:\n"
        '{"score": <integer 0-10>, "reasoning": "<brief reasoning>"}'
    )

    try:
        reply = judge_fn(judge_prompt)
    except Exception as e:
        return {"passed": False, "score": 0.0, "reason": f"Judge error: {e}"}

    # Extract JSON from reply
    text = (reply or "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {"passed": False, "score": 0.0, "reason": f"Could not parse judge reply: {text[:100]}"}
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        return {"passed": False, "score": 0.0, "reason": f"Judge JSON error: {e}"}

    raw = data.get("score", 0)
    try:
        score_int = int(raw)
    except (TypeError, ValueError):
        score_int = 0
    score_int = max(0, min(10, score_int))
    # Normalise to 0-1
    normalised = score_int / 10.0
    passed = score_int >= 7  # threshold: 7/10 = pass
    return {
        "passed": passed,
        "score": normalised,
        "raw_score": score_int,
        "reason": data.get("reasoning", ""),
    }


def grade_task(task: dict, answer: str, judge_fn=None) -> dict:
    """Dispatch to the right grader based on task['grader']."""
    grader = task.get("grader", "exact")
    if grader == "exact":
        return grade_exact(answer, task.get("expected_any", []))
    elif grader == "regex":
        return grade_regex(answer, task.get("expected_regex", ""))
    elif grader == "llm_judge":
        if judge_fn is None:
            return {"passed": False, "score": 0.0,
                    "reason": "No judge function available for llm_judge grader"}
        return grade_llm_judge(
            answer, task.get("rubric", ""), task.get("prompt", ""), judge_fn,
        )
    else:
        return {"passed": False, "score": 0.0, "reason": f"Unknown grader: {grader}"}
