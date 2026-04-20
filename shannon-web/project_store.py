"""
project_store.py — SQLite-backed project & document store for Shannon Web.

Tables:
    projects   — project metadata
    documents  — uploaded/pasted documents (full text stored)
    doc_chunks — chunked text for FTS retrieval
    doc_chunks_fts — FTS5 virtual table for full-text search

Usage:
    store = ProjectStore()            # defaults to ~/.shannon-web/projects.db
    store = ProjectStore("/tmp/test.db")
"""
import json
import re
import sqlite3
import textwrap
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Default DB location ──────────────────────────────────────────────
def _default_db_path() -> Path:
    d = Path.home() / ".shannon-web"
    d.mkdir(parents=True, exist_ok=True)
    return d / "projects.db"


# ── Chunking ─────────────────────────────────────────────────────────
CHUNK_TARGET = 800   # target words per chunk (≈ 1000 tokens)
CHUNK_OVERLAP = 80   # overlap words between consecutive chunks


def chunk_text(text: str, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks.

    Tries to break on paragraph boundaries when possible.
    Returns a list of chunk strings.
    """
    if not text or not text.strip():
        return []

    # Try paragraph-based splitting first
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        # If adding this paragraph would exceed target, flush current chunk
        if current_words and len(current_words) + len(words) > target:
            chunks.append(' '.join(current_words))
            # Keep overlap from the end of current chunk
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(words)

        # If current chunk is already over target, flush
        while len(current_words) > target:
            chunks.append(' '.join(current_words[:target]))
            current_words = current_words[target - overlap:]

    # Flush remaining
    if current_words:
        chunks.append(' '.join(current_words))

    return chunks if chunks else [text.strip()]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word for English, ~1.5 for CJK-heavy."""
    if not text:
        return 0
    words = len(text.split())
    # Count CJK characters (Chinese/Japanese/Korean)
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    return int(words * 0.75 + cjk * 0.5)


# ── Store ─────────────────────────────────────────────────────────────
class ProjectStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # SQLite connections are not safe for concurrent writes from multiple
        # threads even with check_same_thread=False. Serialise all access.
        self._lock = threading.Lock()
        self._init_tables()

    # ── Schema ────────────────────────────────────────────────────────
    def _init_tables(self):
        self._conn.executescript(textwrap.dedent("""\
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                doc_type    TEXT DEFAULT 'other',
                token_count INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS doc_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                project_id  TEXT NOT NULL,
                chunk_idx   INTEGER NOT NULL,
                content     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_project ON doc_chunks(project_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON doc_chunks(doc_id);
        """))

        # FTS5 virtual table (separate statement — can't be in executescript easily)
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts "
                "USING fts5(content, content='doc_chunks', content_rowid='id')"
            )
        except sqlite3.OperationalError:
            pass  # already exists

        # Triggers to keep FTS in sync
        for trigger_sql in [
            """\
            CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
                INSERT INTO doc_chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;""",
            """\
            CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
                INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
            END;""",
            """\
            CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE ON doc_chunks BEGIN
                INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
                INSERT INTO doc_chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;""",
        ]:
            try:
                self._conn.execute(trigger_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    # ── Projects ──────────────────────────────────────────────────────
    def create_project(self, name: str, description: str = "") -> dict:
        pid = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO projects (id, name, description, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (pid, name, description, now, now),
            )
            self._conn.commit()
        return {"id": pid, "name": name, "description": description,
                "created_at": now, "updated_at": now}

    def list_projects(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.*, "
                "  (SELECT COUNT(*) FROM documents WHERE project_id = p.id) AS doc_count, "
                "  (SELECT COALESCE(SUM(token_count), 0) FROM documents WHERE project_id = p.id) AS total_tokens "
                "FROM projects p ORDER BY p.updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT p.*, "
                "  (SELECT COUNT(*) FROM documents WHERE project_id = p.id) AS doc_count, "
                "  (SELECT COALESCE(SUM(token_count), 0) FROM documents WHERE project_id = p.id) AS total_tokens "
                "FROM projects p WHERE p.id = ?", (project_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_project(self, project_id: str, name: str = None, description: str = None) -> Optional[dict]:
        proj = self.get_project(project_id)
        if not proj:
            return None
        name = name if name is not None else proj["name"]
        desc = description if description is not None else proj["description"]
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE projects SET name=?, description=?, updated_at=? WHERE id=?",
                (name, desc, now, project_id),
            )
            self._conn.commit()
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ── Documents ─────────────────────────────────────────────────────
    def add_document(self, project_id: str, title: str, content: str,
                     doc_type: str = "other") -> dict:
        """Add a document, auto-chunk it, and index in FTS5."""
        doc_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        tokens = estimate_tokens(content)
        chunks = chunk_text(content)

        with self._lock:
            self._conn.execute(
                "INSERT INTO documents (id, project_id, title, content, doc_type, token_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, project_id, title, content, doc_type, tokens, now),
            )
            for idx, chunk in enumerate(chunks):
                self._conn.execute(
                    "INSERT INTO doc_chunks (doc_id, project_id, chunk_idx, content) "
                    "VALUES (?, ?, ?, ?)",
                    (doc_id, project_id, idx, chunk),
                )
            self._conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), project_id),
            )
            self._conn.commit()

        return {"id": doc_id, "project_id": project_id, "title": title,
                "doc_type": doc_type, "token_count": tokens,
                "chunk_count": len(chunks), "created_at": now}

    def list_documents(self, project_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, project_id, title, doc_type, token_count, created_at "
                "FROM documents WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ── Context Retrieval (the key feature) ───────────────────────────
    def get_document_index(self, project_id: str) -> str:
        """Return a compact document index for the system prompt.

        Example output:
            ## Project Documents
            1. Design Flow v2.3 (design_flow, ~3200 tokens)
            2. API Spec (system_flow, ~1500 tokens)
        """
        docs = self.list_documents(project_id)
        if not docs:
            return ""
        lines = ["## Project Documents"]
        for i, d in enumerate(docs, 1):
            lines.append(f"{i}. {d['title']} ({d['doc_type']}, ~{d['token_count']} tokens)")
        return '\n'.join(lines)

    def retrieve_context(self, project_id: str, query: str,
                         max_chunks: int = 15, max_tokens: int = 20000) -> str:
        """FTS5 search for chunks relevant to `query` within a project.

        Returns formatted context string ready for system prompt injection.
        Falls back to first N chunks if query is empty or FTS returns nothing.
        """
        chunks: list[dict] = []

        if query.strip():
            # Tokenize query for FTS5 — join with OR for broad matching
            terms = re.findall(r'\w+', query)
            if terms:
                fts_query = ' OR '.join(terms)
                try:
                    with self._lock:
                        rows = self._conn.execute(
                            "SELECT c.content, c.chunk_idx, d.title, d.doc_type, "
                            "       rank "
                            "FROM doc_chunks_fts f "
                            "JOIN doc_chunks c ON c.id = f.rowid "
                            "JOIN documents d ON d.id = c.doc_id "
                            "WHERE c.project_id = ? AND doc_chunks_fts MATCH ? "
                            "ORDER BY rank "
                            "LIMIT ?",
                            (project_id, fts_query, max_chunks),
                        ).fetchall()
                    chunks = [dict(r) for r in rows]
                except sqlite3.OperationalError:
                    pass  # FTS query syntax issue, fall through

        # Fallback: if no FTS results, return first N chunks (chronological)
        if not chunks:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT c.content, c.chunk_idx, d.title, d.doc_type "
                    "FROM doc_chunks c "
                    "JOIN documents d ON d.id = c.doc_id "
                    "WHERE c.project_id = ? "
                    "ORDER BY d.created_at, c.chunk_idx "
                    "LIMIT ?",
                    (project_id, max_chunks),
                ).fetchall()
            chunks = [dict(r) for r in rows]

        if not chunks:
            return ""

        # Build context string, respecting token budget
        lines: list[str] = ["## Relevant Project Context\n"]
        token_count = 0
        for ch in chunks:
            entry = f"### [{ch['title']}] (section {ch['chunk_idx'] + 1})\n{ch['content']}\n"
            entry_tokens = estimate_tokens(entry)
            if token_count + entry_tokens > max_tokens:
                break
            lines.append(entry)
            token_count += entry_tokens

        return '\n'.join(lines)

    def build_project_context(self, project_id: str, user_message: str,
                              max_tokens: int = 20000) -> str:
        """Build the full project context block for injection into system prompt.

        Returns:  document index + relevant chunks (FTS-matched to user_message)
        """
        index = self.get_document_index(project_id)
        relevant = self.retrieve_context(project_id, user_message, max_tokens=max_tokens)

        parts = [p for p in [index, relevant] if p]
        return '\n\n'.join(parts) if parts else ""

    # ── Cleanup ───────────────────────────────────────────────────────
    def close(self):
        self._conn.close()
