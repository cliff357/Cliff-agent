"""Tests for the benchmark module."""
import json
import sys
from pathlib import Path

import pytest

# Ensure shannon-web on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.grader import grade_exact, grade_regex, grade_llm_judge, grade_task
from bench.store import BenchStore
from bench.suites_loader import list_suites, load_suite


# ── Grader ────────────────────────────────────────────────────────────

def test_grade_exact_pass():
    r = grade_exact("The answer is Paris.", ["Paris", "paris"])
    assert r["passed"] is True
    assert r["score"] == 1.0

def test_grade_exact_case_insensitive():
    r = grade_exact("the capital is PARIS", ["Paris"])
    assert r["passed"] is True

def test_grade_exact_fail():
    r = grade_exact("London", ["Paris"])
    assert r["passed"] is False
    assert r["score"] == 0.0

def test_grade_exact_empty_answer():
    r = grade_exact("", ["anything"])
    assert r["passed"] is False

def test_grade_regex_pass():
    r = grade_regex("Pi = 3.14159", r"3\.?14159")
    assert r["passed"] is True

def test_grade_regex_word_boundary():
    r = grade_regex("The answer is 4", r"\b4\b")
    assert r["passed"] is True

def test_grade_regex_no_match():
    r = grade_regex("nothing here", r"\b42\b")
    assert r["passed"] is False

def test_grade_regex_invalid_pattern():
    r = grade_regex("text", "[invalid")
    assert r["passed"] is False

def test_llm_judge_parses_json():
    def fake_judge(prompt):
        return '{"score": 9, "reasoning": "Good answer"}'
    r = grade_llm_judge("The mitochondria is the powerhouse", "rubric", "question", fake_judge)
    assert r["passed"] is True
    assert r["raw_score"] == 9
    assert r["score"] == 0.9

def test_llm_judge_extracts_json_from_extra_text():
    def fake_judge(prompt):
        return 'Here is my judgment: {"score": 7, "reasoning": "ok"} (done)'
    r = grade_llm_judge("x", "y", "q", fake_judge)
    assert r["passed"] is True  # 7 = threshold
    assert r["raw_score"] == 7

def test_llm_judge_low_score_fails():
    def fake_judge(prompt):
        return '{"score": 4, "reasoning": "Wrong"}'
    r = grade_llm_judge("x", "y", "q", fake_judge)
    assert r["passed"] is False
    assert r["raw_score"] == 4

def test_llm_judge_clamps_to_0_10():
    def fake_judge(prompt):
        return '{"score": 99, "reasoning": "out of range"}'
    r = grade_llm_judge("x", "y", "q", fake_judge)
    assert r["raw_score"] == 10

def test_llm_judge_handles_invalid_json():
    def fake_judge(prompt):
        return "not json at all"
    r = grade_llm_judge("x", "y", "q", fake_judge)
    assert r["passed"] is False
    assert r["score"] == 0.0

def test_grade_task_dispatch_exact():
    task = {"grader": "exact", "expected_any": ["yes"]}
    assert grade_task(task, "yes, it is")["passed"] is True

def test_grade_task_dispatch_regex():
    task = {"grader": "regex", "expected_regex": r"\d+"}
    assert grade_task(task, "there are 42")["passed"] is True

def test_grade_task_unknown_grader():
    task = {"grader": "unknown"}
    r = grade_task(task, "any")
    assert r["passed"] is False


# ── Suite loader ──────────────────────────────────────────────────────

def test_list_suites_includes_starter():
    suites = list_suites()
    ids = [s["id"] for s in suites]
    assert "starter" in ids
    starter = next(s for s in suites if s["id"] == "starter")
    assert starter["task_count"] > 0

def test_load_suite_valid():
    s = load_suite("starter")
    assert s["name"]
    assert len(s["tasks"]) > 0
    # Each task has required fields
    for t in s["tasks"]:
        assert "id" in t
        assert "prompt" in t
        assert "grader" in t

def test_load_suite_not_found():
    with pytest.raises(FileNotFoundError):
        load_suite("does_not_exist")

def test_load_suite_blocks_path_traversal():
    # "../" should be stripped; resolves to "etc" which doesn't exist → raise
    with pytest.raises(FileNotFoundError):
        load_suite("../../etc/passwd")


# ── Store ─────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return BenchStore(db_path=tmp_path / "bench.db")

def test_store_create_and_get_run(store):
    run_id = store.create_run(
        suite_id="starter", suite_name="Starter", pattern="single",
        model="gpt-4", provider="openai", total_tasks=3,
    )
    assert run_id
    run = store.get_run(run_id)
    assert run["status"] == "running"
    assert run["total_tasks"] == 3
    assert run["task_results"] == []

def test_store_records_task_results(store):
    run_id = store.create_run(
        suite_id="s", suite_name="S", pattern="single",
        model="m", provider="p", total_tasks=2,
    )
    task = {"id": "t1", "category": "factual", "prompt": "What?"}
    store.record_task_result(run_id, task, "42", {"passed": True, "score": 1.0, "reason": "ok"}, 1.5)
    store.record_task_result(run_id, {"id": "t2", "category": "reasoning", "prompt": "Why?"},
                              "wrong", {"passed": False, "score": 0.0, "reason": "bad"}, 2.0)
    summary = store.finalise_run(run_id)
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["avg_score"] == 0.5
    assert summary["avg_duration"] == 1.75
    assert summary["status"] == "completed"

def test_store_list_runs_ordered(store):
    import time as _time
    ids = []
    for i in range(3):
        rid = store.create_run(
            suite_id="s", suite_name=f"S{i}", pattern="single",
            model="m", provider="p", total_tasks=1,
        )
        ids.append(rid)
        _time.sleep(1.05)  # ensure distinct second-precision timestamps
    runs = store.list_runs()
    # Most-recent first
    assert len(runs) == 3
    assert runs[0]["id"] == ids[-1]

def test_store_delete_run(store):
    rid = store.create_run(
        suite_id="s", suite_name="S", pattern="single", model="m", provider="p",
        total_tasks=1,
    )
    store.record_task_result(rid, {"id": "t1"}, "a", {"passed": True, "score": 1.0}, 1.0)
    assert store.delete_run(rid) is True
    assert store.get_run(rid) is None
    # task_results cascaded
    with store._conn() as c:
        rows = list(c.execute("SELECT * FROM task_results WHERE run_id=?", (rid,)))
    assert rows == []

def test_store_delete_nonexistent(store):
    assert store.delete_run("nonexistent") is False

def test_store_finalise_empty_run(store):
    rid = store.create_run(
        suite_id="s", suite_name="S", pattern="single", model="m", provider="p",
        total_tasks=0,
    )
    summary = store.finalise_run(rid)
    assert summary["passed"] == 0
    assert summary["avg_score"] == 0.0


# ── Runner (integration-ish with a stub agent) ────────────────────────

def test_runner_single_pattern(tmp_path, monkeypatch):
    """Run a tiny suite against a stubbed AIAgent that returns canned answers."""
    from bench import runner as runner_mod

    class StubAgent:
        def __init__(self, *a, **kw): pass
        def run_conversation(self, msg, *a, **kw):
            # Always return "Paris" — passes the capital question
            return {"final_response": "Paris"}

    monkeypatch.setattr(runner_mod, "AIAgent", StubAgent)

    suite = {
        "id": "tiny",
        "name": "Tiny",
        "tasks": [
            {"id": "q1", "category": "factual", "prompt": "Capital of France?",
             "grader": "exact", "expected_any": ["Paris"]},
            {"id": "q2", "category": "factual", "prompt": "Capital of Japan?",
             "grader": "exact", "expected_any": ["Tokyo"]},  # will fail
        ],
    }
    store = runner_mod.BenchStore(db_path=tmp_path / "t.db")
    r = runner_mod.BenchRunner(store=store)
    import queue as _queue
    q = _queue.Queue()
    rid = r.start(suite=suite, pattern="single", model="stub", event_q=q)

    # Drain events (with timeout)
    events = []
    import time as _time
    deadline = _time.time() + 10
    while _time.time() < deadline:
        try:
            ev = q.get(timeout=0.5)
            events.append(ev)
            if ev[0] == "done":
                break
        except Exception:
            continue

    event_types = [e[0] for e in events]
    assert "run_started" in event_types
    assert "run_finished" in event_types
    assert "done" in event_types

    run = store.get_run(rid)
    assert run["status"] == "completed"
    assert run["total_tasks"] == 2
    assert run["passed"] == 1
    assert run["failed"] == 1
