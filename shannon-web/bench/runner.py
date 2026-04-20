"""Benchmark runner — executes a suite against a chosen pattern, streams progress.

Supported patterns:
  - single: one agent produces the answer (baseline)
  - dual_review: Agent A answers, Agent B reviews, loop up to 2 rounds
"""
from __future__ import annotations
import queue
import time
import threading
from typing import Any, Callable, Optional

# NOTE: server.py inserts the project root into sys.path before importing
# this module, so `run_agent` is available without further path hacking.
from run_agent import AIAgent  # type: ignore

from .grader import grade_task
from .store import BenchStore


# Cap concurrent benchmark runs to prevent runaway resource usage when
# users click "Run" repeatedly.
_MAX_CONCURRENT_RUNS = 2
_run_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT_RUNS)


class BenchRunner:
    """Run a suite, emit progress events to a queue, persist results."""

    def __init__(self, store: Optional[BenchStore] = None):
        self.store = store or BenchStore()
        self._cancel_flags: dict[str, threading.Event] = {}

    # ── Public entrypoint ──
    def start(
        self,
        *,
        suite: dict,
        pattern: str,
        model: str,
        provider: str = "",
        base_url: str = "",
        api_key: str = "",
        judge_model: str = "",
        judge_provider: str = "",
        judge_base_url: str = "",
        judge_api_key: str = "",
        note: str = "",
        event_q: queue.Queue,
    ) -> str:
        """Start a benchmark run in a background thread. Returns run_id.

        Raises RuntimeError if the concurrency cap is already reached.
        """
        if not _run_semaphore.acquire(blocking=False):
            raise RuntimeError(
                f"Too many concurrent benchmark runs (max {_MAX_CONCURRENT_RUNS}). "
                "Wait for one to finish or cancel it."
            )

        tasks = suite.get("tasks", [])
        # Default judge to solver if not explicitly set — but the web layer
        # should pass a distinct judge to avoid self-preference bias on
        # llm_judge tasks.
        judge_model = judge_model or model
        judge_provider = judge_provider or provider
        judge_base_url = judge_base_url or base_url
        judge_api_key = judge_api_key or api_key

        run_id = self.store.create_run(
            suite_id=suite.get("id", "unknown"),
            suite_name=suite.get("name", "Unknown"),
            pattern=pattern,
            model=model,
            provider=provider,
            total_tasks=len(tasks),
            note=note,
            meta={
                "base_url": base_url,
                "judge_model": judge_model,
                "judge_provider": judge_provider,
            },
        )
        cancel = threading.Event()
        self._cancel_flags[run_id] = cancel

        def worker():
            try:
                self._run_suite(
                    run_id=run_id,
                    suite=suite,
                    pattern=pattern,
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                    judge_model=judge_model,
                    judge_provider=judge_provider,
                    judge_base_url=judge_base_url,
                    judge_api_key=judge_api_key,
                    event_q=event_q,
                    cancel=cancel,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                event_q.put(("error", {"message": str(e)}))
                self.store.finalise_run(run_id, status="error")
            finally:
                self._cancel_flags.pop(run_id, None)
                _run_semaphore.release()
                event_q.put(("done", {"run_id": run_id}))

        threading.Thread(target=worker, daemon=True).start()
        return run_id

    def cancel(self, run_id: str) -> bool:
        """Request cancellation of an active run."""
        flag = self._cancel_flags.get(run_id)
        if flag:
            flag.set()
            return True
        return False

    # ── Internals ──
    def _make_agent(self, model: str, provider: str, base_url: str, api_key: str) -> AIAgent:
        return AIAgent(
            model=model, provider=provider, base_url=base_url, api_key=api_key,
            platform="cli", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )

    def _ask_single(self, agent: AIAgent, prompt: str) -> str:
        result = agent.run_conversation(prompt)
        return result.get("final_response", "") or ""

    def _ask_dual_review(
        self,
        *,
        prompt: str,
        model: str, provider: str, base_url: str, api_key: str,
        event_q: queue.Queue, task_id: str,
    ) -> str:
        """Solver → Reviewer → optional revision (max 2 rounds)."""
        solver = self._make_agent(model, provider, base_url, api_key)
        answer = self._ask_single(solver, prompt)
        if not answer:
            return ""

        for round_num in range(1, 3):
            reviewer = self._make_agent(model, provider, base_url, api_key)
            review_prompt = (
                "You are a strict reviewer. Evaluate another AI's answer.\n\n"
                f"## Question\n{prompt}\n\n"
                f"## Answer\n{answer}\n\n"
                'Respond with JSON only: {"verdict": "pass" or "revise", "feedback": "..."}'
            )
            review = reviewer.run_conversation(review_prompt).get("final_response", "")
            import json as _json
            verdict, feedback = "pass", ""
            try:
                start = review.find("{"); end = review.rfind("}") + 1
                if start >= 0 and end > start:
                    data = _json.loads(review[start:end])
                    verdict = str(data.get("verdict", "pass")).lower().strip()
                    feedback = data.get("feedback", "")
            except Exception:
                pass
            event_q.put(("review_round", {
                "task_id": task_id, "round": round_num,
                "verdict": verdict, "feedback": feedback[:200],
            }))
            if verdict == "pass":
                break
            # Revise
            revision = (
                f"Your previous answer needs revision.\n\n"
                f"## Original Question\n{prompt}\n\n"
                f"## Previous Answer\n{answer}\n\n"
                f"## Reviewer Feedback\n{feedback}\n\n"
                "Provide an improved answer addressing the feedback."
            )
            answer = solver.run_conversation(revision).get("final_response", "") or answer
        return answer

    def _make_judge_fn(self, model: str, provider: str, base_url: str, api_key: str):
        """Return a callable(prompt: str) -> str using a fresh agent per call."""
        def judge(prompt: str) -> str:
            judge_agent = self._make_agent(model, provider, base_url, api_key)
            result = judge_agent.run_conversation(prompt)
            return result.get("final_response", "") or ""
        return judge

    def _run_suite(
        self,
        *,
        run_id: str,
        suite: dict,
        pattern: str,
        model: str,
        provider: str,
        base_url: str,
        api_key: str,
        judge_model: str,
        judge_provider: str,
        judge_base_url: str,
        judge_api_key: str,
        event_q: queue.Queue,
        cancel: threading.Event,
    ):
        tasks = suite.get("tasks", [])
        judge_fn = self._make_judge_fn(
            judge_model, judge_provider, judge_base_url, judge_api_key,
        )

        event_q.put(("run_started", {
            "run_id": run_id,
            "suite_name": suite.get("name", ""),
            "pattern": pattern,
            "model": model,
            "judge_model": judge_model,
            "total": len(tasks),
        }))

        for idx, task in enumerate(tasks):
            if cancel.is_set():
                self.store.finalise_run(run_id, status="cancelled")
                event_q.put(("cancelled", {"run_id": run_id}))
                return

            task_id = task.get("id", f"task_{idx}")
            event_q.put(("task_started", {
                "run_id": run_id, "index": idx, "total": len(tasks),
                "task_id": task_id, "category": task.get("category", ""),
                "prompt": task.get("prompt", "")[:300],
            }))

            t0 = time.time()
            prompt = task.get("prompt", "")
            try:
                if pattern == "dual_review":
                    answer = self._ask_dual_review(
                        prompt=prompt, model=model, provider=provider,
                        base_url=base_url, api_key=api_key,
                        event_q=event_q, task_id=task_id,
                    )
                else:
                    agent = self._make_agent(model, provider, base_url, api_key)
                    answer = self._ask_single(agent, prompt)
            except Exception as e:
                answer = f"[ERROR: {e}]"

            duration = time.time() - t0
            grade = grade_task(task, answer, judge_fn=judge_fn)
            self.store.record_task_result(run_id, task, answer, grade, duration)

            event_q.put(("task_done", {
                "run_id": run_id, "index": idx, "total": len(tasks),
                "task_id": task_id,
                "passed": grade.get("passed", False),
                "score": grade.get("score", 0.0),
                "raw_score": grade.get("raw_score"),
                "reason": grade.get("reason", ""),
                "answer_preview": (answer or "")[:300],
                "duration": round(duration, 2),
            }))

        summary = self.store.finalise_run(run_id, status="completed")
        event_q.put(("run_finished", {
            "run_id": run_id,
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "total": summary.get("total_tasks", 0),
            "avg_score": summary.get("avg_score", 0.0),
            "avg_duration": summary.get("avg_duration", 0.0),
        }))
