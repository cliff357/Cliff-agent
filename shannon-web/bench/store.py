"""SQLite storage for benchmark run history."""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class BenchStore:
    """Persists benchmark runs + per-task results."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".shannon-web" / "benchmarks.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id           TEXT PRIMARY KEY,
                    suite_id     TEXT NOT NULL,
                    suite_name   TEXT NOT NULL,
                    pattern      TEXT NOT NULL,           -- 'single' | 'dual_review' | ...
                    model        TEXT NOT NULL,
                    provider     TEXT,
                    started_at   INTEGER NOT NULL,
                    finished_at  INTEGER,
                    status       TEXT NOT NULL,           -- 'running' | 'completed' | 'error' | 'cancelled'
                    total_tasks  INTEGER NOT NULL,
                    passed       INTEGER DEFAULT 0,
                    failed       INTEGER DEFAULT 0,
                    avg_score    REAL DEFAULT 0,
                    avg_duration REAL DEFAULT 0,           -- seconds per task
                    note         TEXT,
                    meta_json    TEXT
                );
                CREATE TABLE IF NOT EXISTS task_results (
                    run_id       TEXT NOT NULL,
                    task_id      TEXT NOT NULL,
                    category     TEXT,
                    prompt       TEXT,
                    answer       TEXT,
                    passed       INTEGER NOT NULL,
                    score        REAL NOT NULL,
                    raw_score    INTEGER,
                    reason       TEXT,
                    duration     REAL,
                    started_at   INTEGER,
                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
            """)

    # ── Runs ──
    def create_run(self, *, suite_id: str, suite_name: str, pattern: str,
                   model: str, provider: str, total_tasks: int,
                   note: str = "", meta: Optional[dict] = None) -> str:
        run_id = uuid.uuid4().hex[:16]
        now = int(time.time())
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, suite_id, suite_name, pattern, model, provider, "
                "started_at, status, total_tasks, note, meta_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                (run_id, suite_id, suite_name, pattern, model, provider,
                 now, total_tasks, note, json.dumps(meta or {})),
            )
        return run_id

    def record_task_result(self, run_id: str, task: dict, answer: str,
                           grade: dict, duration: float) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO task_results "
                "(run_id, task_id, category, prompt, answer, passed, score, raw_score, "
                " reason, duration, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, task.get("id", ""), task.get("category", ""),
                    task.get("prompt", ""), answer or "",
                    1 if grade.get("passed") else 0,
                    float(grade.get("score", 0.0)),
                    grade.get("raw_score"),
                    grade.get("reason", ""),
                    duration,
                    int(time.time()),
                ),
            )

    def finalise_run(self, run_id: str, *, status: str = "completed") -> dict:
        with self._conn() as c:
            rows = list(c.execute(
                "SELECT passed, score, duration FROM task_results WHERE run_id=?",
                (run_id,)))
            passed = sum(r["passed"] for r in rows)
            failed = len(rows) - passed
            avg_score = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
            avg_duration = sum(r["duration"] or 0 for r in rows) / len(rows) if rows else 0.0
            now = int(time.time())
            c.execute(
                "UPDATE runs SET finished_at=?, status=?, passed=?, failed=?, "
                "avg_score=?, avg_duration=? WHERE id=?",
                (now, status, passed, failed, avg_score, avg_duration, run_id),
            )
        return self.get_run(run_id) or {}

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return None
            run = dict(row)
            task_rows = c.execute(
                "SELECT * FROM task_results WHERE run_id=? ORDER BY started_at",
                (run_id,),
            ).fetchall()
            run["task_results"] = [dict(r) for r in task_rows]
        return run

    def delete_run(self, run_id: str) -> bool:
        with self._conn() as c:
            c.execute("DELETE FROM task_results WHERE run_id=?", (run_id,))
            cur = c.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return cur.rowcount > 0
