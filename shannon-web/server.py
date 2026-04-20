"""
shannon-web — A minimal web frontend for Hermes Agent.

Run:
    cd shannon-web
    python server.py          # defaults to port 8080
    python server.py --port 9000
"""
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from project_store import ProjectStore

# ── Make sure parent project is importable ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_agent import AIAgent  # noqa: E402
from bench import BenchRunner, BenchStore, load_suite, list_suites  # noqa: E402

# ── Limits ───────────────────────────────────────────────────────────
_MAX_WEB_SESSIONS = 50  # LRU cap — oldest evicted when exceeded

# ── Load Hermes config once at startup ───────────────────────────────
# NOTE: Cached at import time. Edits to ~/.hermes/config.yaml require a
# server restart to take effect.
def _load_hermes_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config()
    except Exception:
        return {}

_HERMES_CFG = _load_hermes_config()

# ── Project store ─────────────────────────────────────────────────────
_project_store = ProjectStore()

_TRIVIAL_PATTERNS = re.compile(
    r"^(hi|hey|hello|yo|sup|hola|howdy|good\s*(morning|afternoon|evening|night)"
    r"|thanks?(\s+you)?|thx|ok(ay)?|bye|see\s+ya|cheers|lol|haha|hmm+|wow"
    r"|yes|no|yep|nope|sure|cool|nice|great|awesome|k|kk"
    r"|你好|嗨|哈囉|早安|晚安|謝謝|掰掰|係|唔係|好|ok啦)[\s!?.~]*$",
    re.IGNORECASE,
)

def _is_substantive_query(message: str) -> bool:
    """Return True if the message looks like it needs document retrieval."""
    msg = message.strip()
    if len(msg) < 3:
        return False
    if _TRIVIAL_PATTERNS.match(msg):
        return False
    # Very short messages with no question-like content
    if len(msg.split()) <= 2 and "?" not in msg:
        return False
    return True
def _resolved_defaults() -> dict:
    """Return {model, provider, base_url} from ~/.hermes/config.yaml."""
    model_cfg = _HERMES_CFG.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model    = model_cfg.get("default") or model_cfg.get("model") or ""
    provider = model_cfg.get("provider") or ""
    base_url = model_cfg.get("base_url") or ""
    return {"model": model, "provider": provider, "base_url": base_url}

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Hermes Web")

# ── Session store ─────────────────────────────────────────────────────
# Each browser session gets its own AIAgent instance so conversation
# history is preserved across messages. Uses an LRU OrderedDict capped at
# _MAX_WEB_SESSIONS to prevent unbounded memory growth.
_sessions: "OrderedDict[str, AIAgent]" = OrderedDict()
_sessions_lock = threading.Lock()

STATIC_DIR = Path(__file__).parent / "static"


def _apply_agent_config(agent: AIAgent, *, model: str = "",
                        provider: str = "", base_url: str = "",
                        api_key: str = "") -> None:
    """Hotswap an agent's model/provider/base_url/api_key in place.

    Rebuilds the underlying OpenAI client if base_url or api_key changed,
    and recomputes api_mode + prompt-caching eligibility.
    Called from both _get_or_create_agent hotswap and /session/<id>/model.
    """
    if model:
        agent.model = model
    if provider:
        agent.provider = provider.strip().lower()
    new_base_url = (base_url or "").strip()
    new_api_key = (api_key or "").strip()

    # Default dummy key for local custom endpoints
    if not new_api_key and agent.provider == "custom" and (new_base_url or agent.base_url):
        new_api_key = "local"

    # Rebuild client if base_url or api_key changed
    need_rebuild = False
    if new_base_url and new_base_url != (agent.base_url or ""):
        agent.base_url = new_base_url  # setter updates _base_url_lower
        need_rebuild = True
    if new_api_key:
        need_rebuild = True

    if need_rebuild:
        try:
            from openai import OpenAI
            key = new_api_key or getattr(agent.client, "api_key", "") or "local"
            url = agent.base_url or None
            agent.client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)
            agent._client_kwargs = {"api_key": key, "base_url": url}
            if new_api_key:
                agent.api_key = new_api_key
        except Exception:
            traceback.print_exc()

    # Recompute api_mode (mirrors AIAgent.__init__)
    p = agent.provider
    bl = agent._base_url_lower
    if p == "openai-codex":
        agent.api_mode = "codex_responses"
    elif not p and "chatgpt.com/backend-api/codex" in bl:
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
    elif p == "anthropic" or (not p and "api.anthropic.com" in bl):
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
    elif bl.rstrip("/").endswith("/anthropic"):
        agent.api_mode = "anthropic_messages"
    else:
        agent.api_mode = "chat_completions"

    if (
        agent.api_mode == "chat_completions"
        and agent.provider != "copilot-acp"
        and not str(agent.base_url or "").lower().startswith("acp://copilot")
        and not str(agent.base_url or "").lower().startswith("acp+tcp://")
        and (agent._is_direct_openai_url() or agent._model_requires_responses_api(agent.model))
    ):
        agent.api_mode = "codex_responses"

    # Recompute prompt-caching eligibility
    is_openrouter = agent._is_openrouter_url()
    is_claude = "claude" in (agent.model or "").lower()
    is_native_anthropic = agent.api_mode == "anthropic_messages" and agent.provider == "anthropic"
    agent._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic


def _make_worker_agent(*, model: str = "", provider: str = "",
                      base_url: str = "", api_key: str = "") -> AIAgent:
    """Construct a fresh AIAgent for ephemeral work (dual-review, benchmark)."""
    eff_api_key = api_key or None
    if not eff_api_key and provider == "custom" and base_url:
        eff_api_key = "local"
    return AIAgent(
        model=model or None,
        provider=provider or None,
        base_url=base_url or None,
        api_key=eff_api_key,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _get_or_create_agent(session_id: str, *, model: str = "",
                         provider: str = "", base_url: str = "",
                         api_key: str = "") -> AIAgent:
    with _sessions_lock:
        if session_id not in _sessions:
            d = _resolved_defaults()
            eff_model    = model    or d["model"]
            eff_provider = provider or d["provider"] or None
            eff_base_url = base_url or d["base_url"] or None

            eff_api_key = api_key or None
            if not eff_api_key and eff_provider == "custom" and eff_base_url:
                eff_api_key = "local"

            _sessions[session_id] = AIAgent(
                model=eff_model,
                provider=eff_provider,
                base_url=eff_base_url,
                api_key=eff_api_key,
                platform="cli",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            # Evict oldest if over cap
            while len(_sessions) > _MAX_WEB_SESSIONS:
                _sessions.popitem(last=False)
        else:
            # Mark as recently used
            _sessions.move_to_end(session_id)
            agent = _sessions[session_id]
            # Hotswap if the client supplied different config
            if (
                (model and model != agent.model)
                or (provider and provider.strip().lower() != (agent.provider or ""))
                or (base_url and base_url != (agent.base_url or ""))
                or api_key
            ):
                _apply_agent_config(
                    agent, model=model, provider=provider,
                    base_url=base_url, api_key=api_key,
                )
        return _sessions[session_id]


# ── Request / Response models ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    review_mode: Optional[bool] = False


class SwitchModelRequest(BaseModel):
    model: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


# ── SSE event helpers ─────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Dual-agent review orchestration ──────────────────────────────────
_MAX_REVIEW_ROUNDS = 2

def _run_dual_agent_review(
    q: queue.Queue,
    user_message: str,
    request,
    *,
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
):
    """
    Orchestrate a solver → reviewer → (optional revision) loop.

    Streams SSE events through `q`:
      - review_phase: {phase, label}   — tells UI which phase we're in
      - delta: {text}                  — streamed text (final answer only)
      - review_verdict: {verdict, feedback}  — reviewer's assessment
      - done: {}                       — finished
    """
    last_verdict = "pass"
    try:
        # ── Phase 1: Solver generates an answer ─────────────────────
        q.put(("review_phase", {"phase": "solving", "label": "🧠 Agent A is thinking…"}))

        solver = _make_worker_agent(model=model, provider=provider,
                                    base_url=base_url, api_key=api_key)
        solver.thinking_callback = lambda t: q.put(("thinking", t))
        solver.reasoning_callback = lambda t: q.put(("reasoning", t))

        solver_result = solver.run_conversation(user_message)
        solver_answer = solver_result.get("final_response", "")

        if not solver_answer:
            q.put(("error", "Solver agent returned empty response"))
            q.put(("done", None))
            return

        q.put(("review_phase", {"phase": "solver_done", "label": "✅ Agent A answered",
                                 "preview": solver_answer[:200]}))

        # ── Phase 2: Reviewer evaluates the answer ──────────────────
        for round_num in range(1, _MAX_REVIEW_ROUNDS + 1):
            q.put(("review_phase", {
                "phase": "reviewing",
                "label": f"🔍 Agent B reviewing (round {round_num})…",
            }))

            reviewer = _make_worker_agent(model=model, provider=provider,
                                          base_url=base_url, api_key=api_key)
            reviewer.thinking_callback = lambda t: q.put(("thinking", t))
            reviewer.reasoning_callback = lambda t: q.put(("reasoning", t))

            review_prompt = (
                "You are a critical reviewer. Your job is to evaluate another AI's answer "
                "for correctness, completeness, and clarity.\n\n"
                f"## Original Question\n{user_message}\n\n"
                f"## Agent's Answer\n{solver_answer}\n\n"
                "## Your Task\n"
                "1. Check for factual errors, logical flaws, or missing information.\n"
                "2. Respond with a JSON object (and nothing else):\n"
                '   {"verdict": "pass" or "revise", "feedback": "your detailed feedback"}\n'
                "3. Only say 'pass' if the answer is accurate and complete.\n"
                "4. If you say 'revise', explain exactly what needs to be fixed."
            )

            review_result = reviewer.run_conversation(review_prompt)
            review_text = review_result.get("final_response", "")

            # Parse reviewer verdict
            verdict, feedback = _parse_review_verdict(review_text)
            last_verdict = verdict

            q.put(("review_verdict", {
                "round": round_num,
                "verdict": verdict,
                "feedback": feedback,
            }))

            if verdict == "pass":
                break

            # ── Phase 3: Solver revises based on feedback ───────────
            q.put(("review_phase", {
                "phase": "revising",
                "label": f"✏️ Agent A revising (round {round_num})…",
            }))

            revision_prompt = (
                f"Your previous answer to this question was reviewed and needs revision.\n\n"
                f"## Original Question\n{user_message}\n\n"
                f"## Your Previous Answer\n{solver_answer}\n\n"
                f"## Reviewer Feedback\n{feedback}\n\n"
                "Please provide an improved answer that addresses all the feedback."
            )

            solver_result = solver.run_conversation(revision_prompt)
            solver_answer = solver_result.get("final_response", "")

            if not solver_answer:
                q.put(("error", "Solver returned empty revision"))
                break

        # ── Stream final answer ─────────────────────────────────────
        if last_verdict != "pass":
            q.put(("review_phase", {
                "phase": "max_rounds",
                "label": f"⚠️ Max {_MAX_REVIEW_ROUNDS} rounds reached — final answer may be unreviewed",
            }))
        q.put(("review_phase", {"phase": "final", "label": "📋 Final reviewed answer"}))
        # Send the final answer as deltas so the UI renders it
        # (chunk it to feel like streaming)
        chunk_size = 20
        words = solver_answer.split(" ")
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i > 0:
                chunk = " " + chunk
            q.put(("delta", chunk))
            time.sleep(0.02)  # slight delay for streaming feel

    except Exception as e:
        q.put(("error", str(e)))
        traceback.print_exc()
    finally:
        q.put(("done", None))


def _parse_review_verdict(text: str) -> tuple:
    """Extract verdict and feedback from reviewer response."""
    text = text.strip()
    # Try to parse JSON from the response
    try:
        # Find JSON in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            verdict = data.get("verdict", "pass").lower().strip()
            feedback = data.get("feedback", "")
            if verdict not in ("pass", "revise"):
                verdict = "pass"
            return verdict, feedback
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: if contains "revise" keyword, treat as revise
    lower = text.lower()
    if "revise" in lower or "incorrect" in lower or "error" in lower or "wrong" in lower:
        return "revise", text
    return "pass", text


def _build_usage_data(agent: AIAgent) -> dict:
    """Extract token usage and context window info from an agent after a turn."""
    data = {}
    # Token usage
    prompt_t = getattr(agent, "session_prompt_tokens", 0) or 0
    comp_t = getattr(agent, "session_completion_tokens", 0) or 0
    data["tokens_used"] = prompt_t + comp_t
    data["prompt_tokens"] = prompt_t
    data["completion_tokens"] = comp_t
    # Context window size
    compressor = getattr(agent, "context_compressor", None)
    if compressor:
        data["context_length"] = getattr(compressor, "context_length", 0) or 0
    else:
        try:
            from agent.model_metadata import get_model_context_length
            data["context_length"] = get_model_context_length(
                agent.model, provider=agent.provider
            )
        except Exception:
            data["context_length"] = 0
    return data


# ── Chat endpoint (SSE streaming) ─────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        agent = _get_or_create_agent(
            session_id,
            model=request.model or "",
            provider=request.provider or "",
            base_url=request.base_url or "",
            api_key=request.api_key or "",
        )
    except (ValueError, RuntimeError) as exc:
        # Return a clean SSE error stream so the UI can display it
        err_msg = str(exc)  # capture before exc goes out of scope
        async def _err_gen():
            yield _sse("session", {"session_id": session_id, "model": request.model or "", "provider": request.provider or ""})
            yield _sse("error", {"error": err_msg})
            yield _sse("done", {})
        return StreamingResponse(_err_gen(), media_type="text/event-stream")

    q: queue.Queue = queue.Queue()

    # ── Callbacks (called from worker thread) ────────────────────────
    def on_delta(text: str):
        q.put(("delta", text))

    def on_tool_start(name: str, preview: str = ""):
        q.put(("tool_start", {"name": name, "preview": preview}))

    def on_tool_complete(name: str, result_preview: str = ""):
        q.put(("tool_done", {"name": name, "preview": result_preview}))

    def on_reasoning(text: str):
        q.put(("reasoning", text))

    def on_thinking(text: str):
        q.put(("thinking", text))

    def on_tool_gen(name: str):
        q.put(("tool_gen", {"name": name}))

    def on_status(event_type: str, message: str):
        q.put(("agent_status", {"text": message}))

    # Attach callbacks to the agent instance before this turn
    agent.stream_delta_callback = on_delta
    agent.tool_start_callback = on_tool_start
    agent.tool_complete_callback = on_tool_complete
    agent.reasoning_callback = on_reasoning
    agent.thinking_callback = on_thinking
    agent.tool_gen_callback = on_tool_gen
    agent.status_callback = on_status

    # ── Run agent in a background thread ─────────────────────────────
    # ── Build project context if a project is active ───────────────
    # NOTE: We prepend context to the user message (not system_message)
    # because AIAgent caches the system prompt across turns for prompt
    # caching.  Per-turn context must go in the user message.
    effective_message = request.message
    context_tokens = 0
    if request.project_id:
        doc_index = _project_store.get_document_index(request.project_id)

        relevant = ""
        if _is_substantive_query(request.message):
            relevant = _project_store.retrieve_context(
                request.project_id, request.message
            )

        if doc_index or relevant:
            ctx_parts = []
            if doc_index:
                ctx_parts.append(f"[Project Documents]\n{doc_index}")
            if relevant:
                ctx_parts.append(
                    "[Retrieved Context]\n"
                    "The following sections from the project documents may be relevant:\n\n"
                    + relevant
                )
            else:
                ctx_parts.append(
                    "[Note: No specific sections were retrieved for this message. "
                    "The project has the documents listed above. "
                    "If the user asks about them, mention what's available.]"
                )
            context_block = "\n\n".join(ctx_parts)
            context_tokens = len(context_block.split())
            effective_message = f"{context_block}\n\n---\n\n{request.message}"

    if request.review_mode:
        def run_review():
            _run_dual_agent_review(
                q, effective_message, request,
                model=agent.model, provider=agent.provider,
                base_url=agent.base_url, api_key=agent.api_key,
            )
        thread = threading.Thread(target=run_review, daemon=True)
        thread.start()
    else:
        def run():
            try:
                agent.run_conversation(
                    effective_message,
                )
            except Exception as e:
                q.put(("error", str(e)))
                traceback.print_exc()
            finally:
                # Send usage/context info before done
                try:
                    usage_data = _build_usage_data(agent)
                    if usage_data:
                        q.put(("usage", usage_data))
                except Exception:
                    pass
                q.put(("done", None))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    # ── SSE generator ─────────────────────────────────────────────────
    async def generate():
        d = _resolved_defaults()
        eff_model    = agent.model    or d["model"]    or ""
        eff_provider = agent.provider or d["provider"] or ""
        yield _sse("session", {"session_id": session_id, "model": eff_model, "provider": eff_provider})
        if context_tokens:
            yield _sse("context_info", {"tokens": context_tokens, "project_id": request.project_id or ""})
        loop = asyncio.get_event_loop()

        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=0.1))
            except queue.Empty:
                # Keep-alive ping
                yield ": ping\n\n"
                await asyncio.sleep(0.05)
                continue

            event, payload = item
            if event == "delta":
                yield _sse("delta", {"text": payload})
            elif event == "reasoning":
                yield _sse("reasoning", {"text": payload})
            elif event == "thinking":
                yield _sse("thinking", {"text": payload})
            elif event == "tool_gen":
                yield _sse("tool_gen", payload)
            elif event == "tool_start":
                yield _sse("tool_start", payload)
            elif event == "tool_done":
                yield _sse("tool_done", payload)
            elif event == "review_phase":
                yield _sse("review_phase", payload)
            elif event == "review_verdict":
                yield _sse("review_verdict", payload)
            elif event == "usage":
                yield _sse("usage", payload)
            elif event == "agent_status":
                yield _sse("agent_status", payload)
            elif event == "error":
                yield _sse("error", {"message": payload})
                break
            elif event == "done":
                yield _sse("done", {})
                break

            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Config info (for UI display before any session) ─────────────────
@app.get("/api/config")
async def get_config():
    d = _resolved_defaults()
    return {
        "model":    d["model"]    or "(auto-detect)",
        "provider": d["provider"] or "(auto)",
        "base_url": d["base_url"] or "",
    }


# ── Session warm-up (pre-create agent on page load) ──────────────────
class WarmUpRequest(BaseModel):
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None

@app.post("/api/session/warmup")
async def session_warmup(request: WarmUpRequest):
    session_id = str(uuid.uuid4())
    try:
        agent = _get_or_create_agent(
            session_id,
            model=request.model or "",
            provider=request.provider or "",
            base_url=request.base_url or "",
            api_key=request.api_key or "",
        )
        ctx_len = 0
        compressor = getattr(agent, "context_compressor", None)
        if compressor:
            ctx_len = getattr(compressor, "context_length", 0) or 0
        return {
            "session_id": session_id,
            "model": agent.model or "",
            "provider": agent.provider or "",
            "context_length": ctx_len,
            "ok": True,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc), "ok": False}, status_code=500)


# ── Session info ─────────────────────────────────────────────────────
@app.get("/api/session/{session_id}/info")
async def session_info(session_id: str):
    with _sessions_lock:
        if session_id not in _sessions:
            return JSONResponse({"error": "session not found"}, status_code=404)
        agent = _sessions[session_id]
    return {"session_id": session_id, "model": agent.model, "provider": agent.provider}


# ── Available models list ─────────────────────────────────────────────

def _get_min_context() -> int:
    """Return MINIMUM_CONTEXT_LENGTH from model_metadata."""
    try:
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        return MINIMUM_CONTEXT_LENGTH
    except Exception:
        return 64_000


# Model name prefix → vendor mapping (longest match first)
_OLLAMA_VENDOR_MAP: list[tuple[str, str]] = [
    ("gemma",       "Google"),
    ("gemini",      "Google"),
    ("qwen",        "Alibaba"),
    ("deepseek",    "DeepSeek"),
    ("glm",         "Zhipu"),
    ("kimi",        "Moonshot"),
    ("minimax",     "MiniMax"),
    ("mistral",     "Mistral"),
    ("ministral",   "Mistral"),
    ("devstral",    "Mistral"),
    ("nemotron",    "NVIDIA"),
    ("llama",       "Meta"),
    ("phi",         "Microsoft"),
    ("gpt-oss",     "OpenAI"),
    ("cogito",      "Deep Cogito"),
    ("rnj",         "Renj"),
]


def _get_vendor(model_name: str) -> str:
    """Return vendor name for an Ollama model name."""
    lower = model_name.lower().split(":")[0]
    for prefix, vendor in _OLLAMA_VENDOR_MAP:
        if lower.startswith(prefix):
            return vendor
    return ""


def _ctx_for_provider_model(model_id: str, provider: str) -> int:
    """Resolve context length for a cloud provider model."""
    try:
        from agent.model_metadata import get_model_context_length
        return get_model_context_length(model_id, provider=provider)
    except Exception:
        return 0


def _scrape_family_tags(family: str) -> list[dict]:
    """Scrape ollama.com/library/<fam>/tags for all size variants.

    Returns list of {id, ctx, param_size, dl_size, installed, vendor}.
    Parses context window, download size, and param size directly from HTML.
    """
    url = f"https://ollama.com/library/{family}/tags"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=8.0) as resp:
        html = resp.read().decode()

    # HTML pattern: <hash></span> \u2022 <dl_size> \u2022 <ctx>K context window
    info_re = re.compile(
        r'[0-9a-f]{12}'
        r'(?:<[^>]*>)*'
        r'\s*(?:\u2022|\u00b7|&bull;)\s*'
        r'([\d.]+\s*(?:GB|MB|TB|KB)|-)\s*'
        r'(?:\u2022|\u00b7|&bull;)\s*'
        r'(\d+(?:\.\d+)?[KMG])\s*context'
    )
    tag_re = re.compile(
        r'href="/library/' + re.escape(family) + r':([^"]+)"'
    )

    # Quantization / format markers to skip
    _QUANT = frozenset(('q2_', 'q3_', 'q4_', 'q5_', 'q6_', 'q8_',
                         'bf16', 'fp4', 'fp8', 'fp16',
                         'mlx', '-it-', 'cloud', 'int4', 'int8', 'mxfp', 'nvfp'))

    results = []
    seen: set[str] = set()
    vendor = _get_vendor(family)

    for tag_m in tag_re.finditer(html):
        tag = tag_m.group(1)
        if tag in seen or tag == 'latest':
            continue
        seen.add(tag)
        if any(q in tag.lower() for q in _QUANT):
            continue

        # Look for info in the next 2000 chars
        region = html[tag_m.end():tag_m.end() + 2000]
        info_m = info_re.search(region)

        dl_size = ""
        ctx = 0
        if info_m:
            dl_raw = info_m.group(1).strip()
            ctx_raw = info_m.group(2).strip()
            dl_size = dl_raw if dl_raw != '-' else ''
            if ctx_raw.endswith('K'):
                ctx = int(float(ctx_raw[:-1]) * 1000)
            elif ctx_raw.endswith('M'):
                ctx = int(float(ctx_raw[:-1]) * 1_000_000)
            elif ctx_raw.endswith('G'):
                ctx = int(float(ctx_raw[:-1]) * 1_000_000_000)

        # Extract param size from tag
        param_size = ""
        tag_base = tag.split("-")[0] if "-" in tag else tag
        ps_m = re.match(r'^(\d+\.?\d*[bBtT])$', tag_base)
        if ps_m:
            param_size = ps_m.group(1).upper()
        elif re.match(r'^e\d+\.?\d*[bBtT]$', tag, re.I):
            param_size = tag.upper()
        elif re.match(r'^(\d+\.?\d*m)$', tag_base, re.I):
            param_size = tag_base.upper()

        results.append({
            "id": f"{family}:{tag}",
            "ctx": ctx,
            "param_size": param_size,
            "dl_size": dl_size,
            "installed": False,
            "vendor": vendor,
        })

    return results


def _scrape_ollama_library() -> list[dict]:
    """Fetch all Ollama library models by scraping tags pages.

    1. Calls /api/tags to get the list of model families.
    2. Scrapes each family's tags page in parallel to find all size variants
       with accurate context windows and download sizes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Step 1: get families from /api/tags
    try:
        req = urllib.request.Request(
            "https://ollama.com/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    families: set[str] = set()
    for m in data.get("models", []):
        name = m.get("name", "")
        if name:
            families.add(name.split(":")[0] if ":" in name else name)

    if not families:
        return []

    # Step 2: scrape tags pages in parallel
    all_models: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            futs = {pool.submit(_scrape_family_tags, fam): fam for fam in families}
            for fut in as_completed(futs, timeout=20):
                try:
                    all_models.extend(fut.result())
                except Exception:
                    pass
    except Exception:
        pass

    return all_models


def _ollama_show(name: str) -> dict:
    """Call Ollama /api/show for one model; return {ctx, param_size, quant, family}."""
    try:
        body = json.dumps({"name": name}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
        details = data.get("details", {})
        model_info = data.get("model_info", {})
        # Context length key follows pattern: <family>.context_length
        ctx = 0
        for k, v in model_info.items():
            if k.endswith(".context_length") and isinstance(v, (int, float)):
                ctx = int(v)
                break
        return {
            "ctx": ctx,
            "param_size": details.get("parameter_size", ""),
            "quant": details.get("quantization_level", ""),
            "family": details.get("family", ""),
        }
    except Exception:
        return {"ctx": 0, "param_size": "", "quant": "", "family": ""}


def _fetch_models_for_session(session_id: Optional[str]) -> dict:
    """Return provider models + local Ollama models grouped by source.

    Each model item is a dict: {id, ctx, param_size?, quant?, installed?}
    """
    provider = ""
    base_url = ""
    if session_id:
        with _sessions_lock:
            ag = _sessions.get(session_id)
        if ag:
            provider = ag.provider or ""
            base_url = ag.base_url or ""

    groups: list[dict] = []

    # ── Provider models ──────────────────────────────────────────────
    try:
        from hermes_cli.models import curated_models_for_provider, fetch_github_model_catalog
        from hermes_cli.auth import resolve_provider_runtime_credentials

        if provider == "copilot":
            try:
                creds = resolve_provider_runtime_credentials("copilot") or {}
                api_key = creds.get("api_key", "")
                catalog = fetch_github_model_catalog(api_key=api_key, timeout=5.0)
                if catalog:
                    items = []
                    for item in catalog:
                        mid = str(item.get("id", "")).strip()
                        if not mid:
                            continue
                        ctx = _ctx_for_provider_model(mid, "copilot")
                        items.append({"id": mid, "ctx": ctx})
                    groups.append({"label": "GitHub Copilot", "models": items,
                                   "provider": "copilot", "base_url": "https://api.githubcopilot.com"})
            except Exception:
                pass
        elif provider and provider != "custom":
            pairs = curated_models_for_provider(provider)
            if pairs:
                items = []
                for mid, _ in pairs:
                    if not mid:
                        continue
                    ctx = _ctx_for_provider_model(mid, provider)
                    items.append({"id": mid, "ctx": ctx})
                groups.append({"label": f"{provider.capitalize()} models", "models": items,
                               "provider": provider, "base_url": base_url})
        elif base_url:
            try:
                url = base_url.rstrip("/") + "/models"
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    data = json.loads(resp.read().decode())
                items = []
                for entry in data.get("data", []):
                    mid = str(entry.get("id", "")).strip()
                    if not mid:
                        continue
                    items.append({"id": mid, "ctx": 0})
                if items:
                    groups.append({"label": "Custom endpoint", "models": items,
                                   "provider": "custom", "base_url": base_url})
            except Exception:
                pass
    except ImportError:
        pass

    # ── Local Ollama models (installed) ───────────────────────────────
    installed_names: set = set()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
        local_items = []
        for m in data.get("models", []):
            name = m.get("name", "")
            if not name:
                continue
            installed_names.add(name)
            info = _ollama_show(name)
            local_items.append({
                "id": name,
                "ctx": info["ctx"],
                "param_size": info["param_size"],
                "quant": info["quant"],
                "installed": True,
                "vendor": _get_vendor(name),
            })
        if local_items:
            groups.append({"label": "Ollama (installed)", "models": local_items,
                           "provider": "custom", "base_url": "http://localhost:11434/v1"})
    except Exception:
        pass

    # ── Ollama library (downloadable) ─────────────────────────────────
    try:
        lib_models = _scrape_ollama_library()
        # Filter out already-installed models
        downloadable = [m for m in lib_models if m["id"] not in installed_names]
        if downloadable:
            # Group by vendor
            from collections import OrderedDict
            by_vendor: OrderedDict[str, list] = OrderedDict()
            for m in downloadable:
                v = m.get("vendor") or "Other"
                by_vendor.setdefault(v, []).append(m)
            for vendor, models in by_vendor.items():
                groups.append({"label": f"{vendor}", "models": models,
                               "provider": "custom", "base_url": "http://localhost:11434/v1",
                               "is_library": True})
    except Exception:
        pass

    return {
        "groups": groups,
        "provider": provider,
        "base_url": base_url,
        "minimum_context": _get_min_context(),
    }


@app.get("/api/models")
async def list_models(session_id: Optional[str] = None):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: _fetch_models_for_session(session_id))
    return JSONResponse(result)


# ── Cloud providers (paid API services) ───────────────────────────────

# Provider registry — id, display name, env var for API key, base_url
_CLOUD_PROVIDERS = [
    {"id": "copilot",   "name": "GitHub Copilot",  "env": "COPILOT_GITHUB_TOKEN", "base_url": "https://api.githubcopilot.com", "icon": "🐙"},
    {"id": "anthropic",  "name": "Anthropic",       "env": "ANTHROPIC_API_KEY",    "base_url": "https://api.anthropic.com", "icon": "🔮"},
    {"id": "openai",     "name": "OpenAI",          "env": "OPENAI_API_KEY",       "base_url": "https://api.openai.com/v1", "icon": "🤖"},
    {"id": "google",     "name": "Google Gemini",   "env": "GEMINI_API_KEY",       "base_url": "", "icon": "💎"},
    {"id": "deepseek",   "name": "DeepSeek",        "env": "DEEPSEEK_API_KEY",     "base_url": "https://api.deepseek.com/v1", "icon": "🔍"},
    {"id": "xai",        "name": "xAI (Grok)",      "env": "XAI_API_KEY",          "base_url": "https://api.x.ai/v1", "icon": "⚡"},
    {"id": "groq",       "name": "Groq",            "env": "GROQ_API_KEY",         "base_url": "https://api.groq.com/openai/v1", "icon": "🚀"},
    {"id": "openrouter", "name": "OpenRouter",      "env": "OPENROUTER_API_KEY",   "base_url": "https://openrouter.ai/api/v1", "icon": "🌐"},
]


def _check_provider_has_key(provider_id: str) -> bool:
    """Check if a cloud provider has an API key configured (env or hermes auth)."""
    import os
    prov = next((p for p in _CLOUD_PROVIDERS if p["id"] == provider_id), None)
    if not prov:
        return False
    # Check env var directly
    if os.getenv(prov["env"]):
        return True
    # Check hermes auth system
    try:
        from hermes_cli.auth import resolve_provider
        # For copilot, check dedicated auth
        if provider_id == "copilot":
            try:
                from hermes_cli.copilot_auth import resolve_copilot_token
                token, _ = resolve_copilot_token()
                return bool(token)
            except Exception:
                pass
        return False
    except ImportError:
        return False


def _fetch_cloud_provider_models(provider_id: str, user_api_key: str = "") -> list[dict]:
    """Fetch models for a specific cloud provider.

    If ``user_api_key`` is provided (from the request body), it is passed
    directly to the provider API without mutating ``os.environ`` — which
    would not be thread-safe under FastAPI concurrency.
    """
    try:
        from hermes_cli.models import curated_models_for_provider, fetch_github_model_catalog

        if provider_id == "copilot":
            token = user_api_key
            if not token:
                try:
                    from hermes_cli.copilot_auth import resolve_copilot_token
                    token, _ = resolve_copilot_token()
                except Exception:
                    token = ""
            if token:
                try:
                    catalog = fetch_github_model_catalog(api_key=token, timeout=5.0)
                    if catalog:
                        items = []
                        for item in catalog:
                            mid = str(item.get("id", "")).strip()
                            if not mid:
                                continue
                            ctx = _ctx_for_provider_model(mid, "copilot")
                            items.append({"id": mid, "ctx": ctx})
                        return items
                except Exception:
                    pass

        pairs = curated_models_for_provider(provider_id)
        if pairs:
            items = []
            for mid, _ in pairs:
                if not mid:
                    continue
                ctx = _ctx_for_provider_model(mid, provider_id)
                items.append({"id": mid, "ctx": ctx})
            return items
    except ImportError:
        pass
    return []


class CloudProviderConnectRequest(BaseModel):
    provider_id: str
    api_key: Optional[str] = None


@app.get("/api/cloud-providers")
async def list_cloud_providers():
    """Return list of cloud providers with connection status."""
    result = []
    for prov in _CLOUD_PROVIDERS:
        has_key = _check_provider_has_key(prov["id"])
        result.append({
            "id": prov["id"],
            "name": prov["name"],
            "icon": prov["icon"],
            "base_url": prov["base_url"],
            "connected": has_key,
        })
    return JSONResponse({"providers": result})


@app.post("/api/cloud-providers/{provider_id}/models")
async def cloud_provider_models(provider_id: str, req: Request):
    """Fetch models for a cloud provider. Optionally accepts {api_key} in body."""
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass

    prov = next((p for p in _CLOUD_PROVIDERS if p["id"] == provider_id), None)
    if not prov:
        return JSONResponse({"error": f"Unknown provider: {provider_id}"}, status_code=404)

    user_key = (body.get("api_key") or "").strip()
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(
        None, lambda: _fetch_cloud_provider_models(provider_id, user_key)
    )
    return JSONResponse({
        "provider": provider_id,
        "base_url": prov["base_url"],
        "models": models,
    })


# ── Ollama pull (streaming — the only supported path) ───────────────
class OllamaPullRequest(BaseModel):
    name: str


# ── Streaming pull with progress ─────────────────────────────────────
@app.post("/api/ollama/pull/stream")
async def ollama_pull_stream(req: OllamaPullRequest):
    """Stream Ollama pull progress as SSE events.

    Emits events:
      - progress: {status, total, completed, percent}
      - done:     {ok, status}
      - error:    {message}
    """
    async def generate():
        import threading
        q: queue.Queue = queue.Queue()

        def pull_worker():
            try:
                body = json.dumps({"name": req.name, "stream": True}).encode()
                request = urllib.request.Request(
                    "http://localhost:11434/api/pull",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=1800) as resp:
                    for line in resp:
                        line = line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        q.put(("progress", obj))
                        if obj.get("status") in ("success",) or obj.get("error"):
                            break
                q.put(("done", {"ok": True, "status": "success"}))
            except Exception as e:
                q.put(("error", {"message": str(e)}))
            finally:
                q.put(("__end__", None))

        threading.Thread(target=pull_worker, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=0.25))
            except queue.Empty:
                yield ": ping\n\n"
                continue
            event, data = item
            if event == "__end__":
                break
            yield _sse(event, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Benchmark endpoints ──────────────────────────────────────────────
_bench_store = BenchStore()
_bench_runner = BenchRunner(store=_bench_store)


@app.get("/api/bench/suites")
async def bench_list_suites():
    return JSONResponse(list_suites())


@app.get("/api/bench/suites/{suite_id}")
async def bench_get_suite(suite_id: str):
    try:
        return JSONResponse(load_suite(suite_id))
    except FileNotFoundError:
        return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/bench/runs")
async def bench_list_runs(limit: int = 50):
    return JSONResponse(_bench_store.list_runs(limit))


@app.get("/api/bench/runs/{run_id}")
async def bench_get_run(run_id: str):
    run = _bench_store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(run)


@app.delete("/api/bench/runs/{run_id}")
async def bench_delete_run(run_id: str):
    ok = _bench_store.delete_run(run_id)
    return JSONResponse({"ok": ok})


class BenchRunRequest(BaseModel):
    suite_id: str
    pattern: str = "single"   # 'single' | 'dual_review'
    model: str = ""
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    note: str = ""
    # Judge config — defaults to solver model when empty, but users can
    # pick a different model to avoid self-preference bias in llm_judge.
    judge_model: str = ""
    judge_provider: str = ""
    judge_base_url: str = ""
    judge_api_key: str = ""


@app.post("/api/bench/run")
async def bench_run(req: BenchRunRequest):
    """Start a benchmark run; stream progress as SSE."""
    try:
        suite = load_suite(req.suite_id)
    except FileNotFoundError:
        return JSONResponse({"error": "suite not found"}, status_code=404)

    # Resolve model from web config defaults if not provided
    model = req.model.strip()
    provider = (req.provider or "").strip().lower()
    base_url = req.base_url.strip()
    api_key = req.api_key.strip()
    if not model:
        d = _resolved_defaults()
        model = d.get("model", "") or model
        provider = provider or d.get("provider", "")
        base_url = base_url or d.get("base_url", "")

    # Judge defaults to solver if caller didn't override
    judge_model = (req.judge_model or "").strip() or model
    judge_provider = (req.judge_provider or "").strip().lower() or provider
    judge_base_url = (req.judge_base_url or "").strip() or base_url
    judge_api_key = (req.judge_api_key or "").strip() or api_key

    q: queue.Queue = queue.Queue()
    try:
        run_id = _bench_runner.start(
            suite=suite, pattern=req.pattern,
            model=model, provider=provider,
            base_url=base_url, api_key=api_key,
            judge_model=judge_model, judge_provider=judge_provider,
            judge_base_url=judge_base_url, judge_api_key=judge_api_key,
            note=req.note, event_q=q,
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=429)

    async def generate():
        yield _sse("run_id", {"run_id": run_id})
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=0.5))
            except queue.Empty:
                yield ": ping\n\n"
                continue
            event, data = item
            yield _sse(event, data)
            if event == "done":
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/bench/runs/{run_id}/cancel")
async def bench_cancel(run_id: str):
    ok = _bench_runner.cancel(run_id)
    return JSONResponse({"ok": ok})


# ── Agent dashboard stats ────────────────────────────────────────────
def _collect_dashboard_stats() -> dict:
    """Synchronous worker — called via run_in_executor to avoid blocking the loop."""
    hermes_home = Path.home() / ".hermes"
    stats: dict = {"hermes_home": str(hermes_home)}

    # ── Skills ──
    skills_dir = hermes_home / "skills"
    skills: list[dict] = []
    if skills_dir.exists():
        for p in sorted(skills_dir.iterdir()):
            if not p.is_dir():
                continue
            desc = ""
            readme = p / "SKILL.md"
            if not readme.exists():
                readme = p / "README.md"
            if readme.exists():
                try:
                    head = readme.read_text(errors="ignore")[:400]
                    # first non-heading, non-empty line
                    for ln in head.splitlines():
                        ln = ln.strip()
                        if ln and not ln.startswith("#") and not ln.startswith("---"):
                            desc = ln[:140]
                            break
                except Exception:
                    pass
            skills.append({"name": p.name, "description": desc})
    stats["skills"] = skills

    # ── Memories ──
    memories_dir = hermes_home / "memories"
    memory_items: list[dict] = []
    if memories_dir.exists():
        for p in sorted(memories_dir.glob("**/*")):
            if p.is_file() and p.suffix in (".md", ".txt", ".json"):
                try:
                    text = p.read_text(errors="ignore")[:160]
                except Exception:
                    text = ""
                memory_items.append({
                    "name": p.name,
                    "path": str(p.relative_to(memories_dir)),
                    "size": p.stat().st_size,
                    "preview": text.replace("\n", " ").strip()[:140],
                })
    stats["memories"] = memory_items

    # ── Sessions (from ~/.hermes/sessions/) ──
    sessions_dir = hermes_home / "sessions"
    sessions_count = 0
    recent_sessions: list[dict] = []
    if sessions_dir.exists():
        entries = [p for p in sessions_dir.iterdir() if p.is_dir() or p.suffix == ".json"]
        sessions_count = len(entries)
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in entries[:10]:
            recent_sessions.append({
                "name": p.name,
                "mtime": int(p.stat().st_mtime),
            })
    stats["sessions"] = {"count": sessions_count, "recent": recent_sessions}

    # ── Tools (from registry) ──
    tools_list: list[dict] = []
    try:
        from tools.registry import registry as _tools_registry
        for name in sorted(_tools_registry.list_tools() if hasattr(_tools_registry, "list_tools") else []):
            tools_list.append({"name": name})
    except Exception:
        # Fallback: count tool files
        tools_path = Path(__file__).parent.parent / "tools"
        if tools_path.exists():
            for f in sorted(tools_path.glob("*_tool.py")):
                tools_list.append({"name": f.stem})
    stats["tools"] = tools_list

    # ── Active web sessions ──
    with _sessions_lock:
        stats["web_sessions"] = [
            {
                "id": sid,
                "model": getattr(ag, "model", ""),
                "provider": getattr(ag, "provider", ""),
                "turns": len(getattr(ag, "conversation_history", []) or []),
            }
            for sid, ag in _sessions.items()
        ]

    # ── Projects ──
    try:
        projects = _project_store.list_projects()
        stats["projects"] = [
            {"id": p["id"], "name": p["name"],
             "docs": p.get("doc_count", 0), "tokens": p.get("total_tokens", 0)}
            for p in projects
        ]
    except Exception:
        stats["projects"] = []

    # ── Config summary ──
    cfg_path = hermes_home / "config.yaml"
    stats["config_present"] = cfg_path.exists()
    stats["config_size"] = cfg_path.stat().st_size if cfg_path.exists() else 0

    return stats


@app.get("/api/dashboard")
async def dashboard_stats():
    """Aggregated agent stats (filesystem IO → executor)."""
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _collect_dashboard_stats)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse(data)


# ── Tool availability status ─────────────────────────────────────────
def _collect_tool_status() -> dict:
    """Return per-tool availability + missing env vars.

    This lets the UI show users WHY a tool is unavailable (e.g. web_search
    needs PARALLEL_API_KEY) so they understand why the agent can't do X.
    """
    try:
        from tools.registry import registry
        import model_tools
        model_tools._discover_tools()
    except Exception as e:
        return {"error": f"tool registry unavailable: {e}", "tools": []}

    categories = {
        "web":     ("🌐 Web search",     ["web_search", "web_extract"]),
        "browser": ("🖥️ Browser",        ["browser_navigate", "browser_click",
                                           "browser_type", "browser_snapshot",
                                           "browser_vision", "browser_console"]),
        "file":    ("📁 Files",          ["read_file", "write_file", "edit_file",
                                           "search_files", "patch"]),
        "shell":   ("💻 Terminal",       ["terminal", "execute_code"]),
        "delegate":("🤝 Subagent",       ["delegate"]),
        "mcp":     ("🔌 MCP",            ["mcp"]),
        "memory":  ("🧠 Memory",         ["memory", "todo"]),
    }

    # Build a reverse lookup so uncategorised tools land in "other"
    categorised = set()
    for _, (_, names) in categories.items():
        categorised.update(names)

    tools_out: list[dict] = []
    for name, entry in sorted(registry._tools.items()):
        try:
            avail = entry.check_fn() if entry.check_fn else True
        except Exception:
            avail = False
        missing = [e for e in (entry.requires_env or []) if not os.getenv(e)]
        # Resolve category
        cat = "other"
        for key, (_, names) in categories.items():
            if name in names:
                cat = key
                break
        tools_out.append({
            "name": name,
            "available": bool(avail),
            "missing_env": missing,
            "category": cat,
            "toolset": getattr(entry, "toolset", "") or "",
        })

    # Summary per category
    summary: list[dict] = []
    for key, (label, names) in categories.items():
        cat_tools = [t for t in tools_out if t["category"] == key]
        if not cat_tools:
            continue
        avail_count = sum(1 for t in cat_tools if t["available"])
        all_missing: list[str] = []
        for t in cat_tools:
            for env in t["missing_env"]:
                if env not in all_missing:
                    all_missing.append(env)
        summary.append({
            "key": key,
            "label": label,
            "total": len(cat_tools),
            "available": avail_count,
            "ready": avail_count == len(cat_tools),
            "missing_env": all_missing if avail_count < len(cat_tools) else [],
        })

    return {
        "tools": tools_out,
        "summary": summary,
        "total": len(tools_out),
        "available": sum(1 for t in tools_out if t["available"]),
    }


@app.get("/api/tools/status")
async def tools_status():
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _collect_tool_status)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse(data)


# ── MCP server management (self-extension) ───────────────────────────
# Hermes agents extend themselves by registering MCP servers. These
# endpoints let the UI add / remove servers without hand-editing config.yaml.

class MCPServerAddRequest(BaseModel):
    name: str
    # stdio transport
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict] = None
    # http transport
    url: Optional[str] = None
    headers: Optional[dict] = None
    # common
    timeout: Optional[float] = None
    enabled: Optional[bool] = True


def _mcp_load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _mcp_save_servers(servers: dict) -> None:
    from hermes_cli.config import load_config, save_config
    cfg = load_config() or {}
    cfg["mcp_servers"] = servers
    save_config(cfg)


@app.get("/api/mcp/servers")
async def mcp_list_servers():
    """List configured MCP servers + live connection status."""
    def _work():
        cfg = _mcp_load_config()
        configured = cfg.get("mcp_servers") or {}
        if not isinstance(configured, dict):
            configured = {}
        status_list: list[dict] = []
        try:
            from tools.mcp_tool import get_mcp_status
            status_list = get_mcp_status() or []
        except Exception:
            status_list = []
        status_by_name = {s.get("name"): s for s in status_list}
        out: list[dict] = []
        for name, scfg in configured.items():
            s = status_by_name.get(name, {}) or {}
            out.append({
                "name": name,
                "enabled": scfg.get("enabled", True) if isinstance(scfg, dict) else True,
                "transport": "http" if isinstance(scfg, dict) and scfg.get("url") else "stdio",
                "command": scfg.get("command") if isinstance(scfg, dict) else None,
                "args": scfg.get("args") if isinstance(scfg, dict) else None,
                "url": scfg.get("url") if isinstance(scfg, dict) else None,
                "connected": bool(s.get("connected")),
                "tool_count": s.get("tool_count", 0),
                "error": s.get("error", ""),
            })
        return {"servers": out}

    loop = asyncio.get_event_loop()
    try:
        return JSONResponse(await loop.run_in_executor(None, _work))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/mcp/servers")
async def mcp_add_server(req: MCPServerAddRequest):
    """Add (or replace) an MCP server definition in ~/.hermes/config.yaml."""
    if not req.name:
        return JSONResponse({"error": "name required"}, status_code=400)
    if not (req.command or req.url):
        return JSONResponse({"error": "either command or url required"},
                            status_code=400)

    def _work():
        cfg = _mcp_load_config()
        servers = cfg.get("mcp_servers") or {}
        if not isinstance(servers, dict):
            servers = {}
        scfg: dict = {}
        if req.command:
            scfg["command"] = req.command
            if req.args:
                scfg["args"] = list(req.args)
        if req.url:
            scfg["url"] = req.url
            if req.headers:
                scfg["headers"] = dict(req.headers)
        if req.env:
            scfg["env"] = dict(req.env)
        if req.timeout is not None:
            scfg["timeout"] = float(req.timeout)
        if req.enabled is False:
            scfg["enabled"] = False
        servers[req.name] = scfg
        _mcp_save_servers(servers)
        # Attempt live registration so agent can use it immediately
        registered: list[str] = []
        try:
            from tools.mcp_tool import register_mcp_servers
            registered = register_mcp_servers({req.name: scfg}) or []
        except Exception as e:
            return {"ok": True, "saved": True, "live_register_error": str(e)}
        return {"ok": True, "saved": True, "registered_tools": registered}

    loop = asyncio.get_event_loop()
    try:
        return JSONResponse(await loop.run_in_executor(None, _work))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/mcp/servers/{name}")
async def mcp_remove_server(name: str):
    def _work():
        cfg = _mcp_load_config()
        servers = cfg.get("mcp_servers") or {}
        if not isinstance(servers, dict) or name not in servers:
            return {"ok": False, "error": "not found"}
        servers.pop(name, None)
        _mcp_save_servers(servers)
        return {"ok": True}

    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _work)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    if not data.get("ok"):
        return JSONResponse(data, status_code=404)
    return JSONResponse(data)


# Curated starter list of well-known MCP servers — one-click add targets.
_MCP_CURATED = [
    {
        "name": "fetch",
        "description": "Fetch and convert any URL to markdown. Zero setup, no API key.",
        "why": "Lets the agent read any webpage — drop-in web browsing.",
        "config": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "tags": ["web", "free"],
    },
    {
        "name": "filesystem",
        "description": "Read/write files outside the workspace.",
        "why": "Gives the agent access to additional directories you pick.",
        "config": {"command": "npx", "args": ["-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(Path.home())]},
        "tags": ["files", "free"],
    },
    {
        "name": "duckduckgo",
        "description": "DuckDuckGo search without any API key.",
        "why": "Free web search fallback when you don't have Parallel/Tavily keys.",
        "config": {"command": "uvx", "args": ["duckduckgo-mcp-server"]},
        "tags": ["web", "free", "search"],
    },
    {
        "name": "memory",
        "description": "Persistent knowledge graph that survives across sessions.",
        "why": "Agent can remember facts long-term.",
        "config": {"command": "npx", "args": ["-y",
                    "@modelcontextprotocol/server-memory"]},
        "tags": ["memory", "free"],
    },
    {
        "name": "sqlite",
        "description": "Query any SQLite database.",
        "why": "Analyse local DBs without writing Python.",
        "config": {"command": "uvx", "args": ["mcp-server-sqlite", "--db-path",
                    str(Path.home() / ".hermes" / "memory.db")]},
        "tags": ["data", "free"],
    },
]


@app.get("/api/mcp/curated")
async def mcp_curated():
    """Return curated starter list of well-known MCP servers."""
    return JSONResponse({"servers": _MCP_CURATED})


# ── Skills management (self-extension) ───────────────────────────────
@app.get("/api/skills")
async def skills_list():
    """List all installed skills from ~/.hermes/skills/."""
    def _work():
        skills_dir = Path.home() / ".hermes" / "skills"
        out: list[dict] = []
        if not skills_dir.exists():
            return {"skills": [], "skills_dir": str(skills_dir)}
        for p in sorted(skills_dir.iterdir()):
            if not p.is_dir():
                continue
            desc = ""
            tags: list[str] = []
            readme = p / "SKILL.md"
            if not readme.exists():
                readme = p / "README.md"
            if readme.exists():
                try:
                    head = readme.read_text(errors="ignore")[:1000]
                    for ln in head.splitlines():
                        ln = ln.strip()
                        if ln and not ln.startswith("#") and not ln.startswith("---"):
                            desc = ln[:200]
                            break
                except Exception:
                    pass
            out.append({
                "name": p.name,
                "description": desc,
                "path": str(p),
                "files": len(list(p.rglob("*"))),
            })
        return {"skills": out, "skills_dir": str(skills_dir)}

    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(None, _work))


@app.get("/api/skills/search")
async def skills_search(q: str = "", limit: int = 20):
    """Search skills hub (ClawHub + GitHub taps)."""
    def _work():
        try:
            # Minimal search — use the registry sources directly
            from hermes_cli.skill_registry import get_all_sources
            sources = get_all_sources()
        except Exception as e:
            return {"error": f"skill registry unavailable: {e}", "results": []}
        results: list[dict] = []
        query = (q or "").lower().strip()
        for src in sources:
            try:
                entries = src.list_entries() if hasattr(src, "list_entries") else []
            except Exception:
                entries = []
            for e in entries or []:
                name = (getattr(e, "name", None)
                        or (e.get("name") if isinstance(e, dict) else None) or "")
                desc = (getattr(e, "description", None)
                        or (e.get("description") if isinstance(e, dict) else None) or "")
                if query and query not in name.lower() and query not in desc.lower():
                    continue
                results.append({
                    "name": name,
                    "description": desc,
                    "source": getattr(src, "name", None) or str(src),
                })
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        return {"results": results}

    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(None, _work))


class SkillInstallRequest(BaseModel):
    identifier: str  # e.g. "pdf-reader" or "user/repo#skill-name"


@app.post("/api/skills/install")
async def skills_install(req: SkillInstallRequest):
    """Install a skill from the hub."""
    def _work():
        try:
            from hermes_cli.skills_hub import do_install
        except Exception as e:
            return {"ok": False, "error": f"skills_hub unavailable: {e}"}
        # Run headless — capture exceptions as error
        try:
            from rich.console import Console
            do_install(req.identifier, force=True, console=Console(quiet=True))
            return {"ok": True, "name": req.identifier}
        except SystemExit:
            return {"ok": False, "error": "install aborted"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(None, _work))


@app.delete("/api/skills/{name}")
async def skills_uninstall(name: str):
    """Remove a skill from ~/.hermes/skills/."""
    def _work():
        skill_path = Path.home() / ".hermes" / "skills" / name
        if not skill_path.exists() or not skill_path.is_dir():
            return {"ok": False, "error": "not found"}
        import shutil
        try:
            shutil.rmtree(skill_path)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _work)
    if not data.get("ok"):
        return JSONResponse(data, status_code=404 if data.get("error") == "not found" else 500)
    return JSONResponse(data)


# ── Hotswap model mid-session ─────────────────────────────────────────
@app.patch("/api/session/{session_id}/model")
async def switch_model(session_id: str, req: SwitchModelRequest):
    """Switch model on an existing agent, preserving conversation history."""
    with _sessions_lock:
        if session_id not in _sessions:
            return JSONResponse({"error": "session not found"}, status_code=404)
        agent = _sessions[session_id]
        _sessions.move_to_end(session_id)

    _apply_agent_config(
        agent,
        model=req.model or "",
        provider=req.provider or "",
        base_url=req.base_url or "",
        api_key=req.api_key or "",
    )
    return {"ok": True, "model": agent.model, "provider": agent.provider,
            "api_mode": agent.api_mode}


# ── Remote Ollama proxy ───────────────────────────────────────────────

def _normalize_ollama_host(host: str) -> str:
    """Normalize host input to http://host:port format."""
    host = host.strip().rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    from urllib.parse import urlparse
    parsed = urlparse(host)
    if not parsed.port:
        host = f"{parsed.scheme}://{parsed.hostname}:11434"
    return host


def _remote_ollama_request(host: str, path: str, token: str,
                           method: str = "GET", body: dict = None,
                           timeout: float = 8.0) -> dict:
    """Make an authenticated request to a remote Ollama instance."""
    base = _normalize_ollama_host(host)
    url = f"{base}{path}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data_bytes = None
    if body is not None:
        data_bytes = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
        if method == "GET":
            method = "POST"
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _remote_ollama_show(host: str, token: str, name: str) -> dict:
    """Call /api/show on a remote Ollama instance."""
    try:
        data = _remote_ollama_request(host, "/api/show", token,
                                      method="POST", body={"name": name}, timeout=5.0)
        details = data.get("details", {})
        model_info = data.get("model_info", {})
        ctx = 0
        for k, v in model_info.items():
            if k.endswith(".context_length") and isinstance(v, (int, float)):
                ctx = int(v)
                break
        return {
            "ctx": ctx,
            "param_size": details.get("parameter_size", ""),
            "quant": details.get("quantization_level", ""),
            "family": details.get("family", ""),
        }
    except Exception:
        return {"ctx": 0, "param_size": "", "quant": "", "family": ""}


def _fetch_remote_ollama_models(host: str, token: str) -> dict:
    """Fetch installed + library models for a remote Ollama instance."""
    base = _normalize_ollama_host(host)
    api_key = token or "remote"
    base_url = base.rstrip("/") + "/v1"

    try:
        data = _remote_ollama_request(host, "/api/tags", token, timeout=5.0)
    except Exception as e:
        return {"error": str(e), "groups": [], "minimum_context": _get_min_context()}

    groups: list[dict] = []
    installed_names: set[str] = set()
    items = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if not name:
            continue
        installed_names.add(name)
        info = _remote_ollama_show(host, token, name)
        items.append({
            "id": name,
            "ctx": info["ctx"],
            "param_size": info["param_size"],
            "quant": info["quant"],
            "installed": True,
            "vendor": _get_vendor(name),
        })

    if items:
        items.sort(key=lambda m: (m.get("vendor") or "zzz", m["id"]))
        groups.append({
            "label": "Remote Ollama (installed)",
            "models": items,
            "provider": "custom",
            "base_url": base_url,
            "api_key": api_key,
        })

    # Library models (for pulling to remote)
    try:
        lib_models = _scrape_ollama_library()
        downloadable = [m for m in lib_models if m["id"] not in installed_names]
        if downloadable:
            from collections import OrderedDict
            by_vendor: OrderedDict[str, list] = OrderedDict()
            for m in downloadable:
                v = m.get("vendor") or "Other"
                by_vendor.setdefault(v, []).append(m)
            for vendor, models in by_vendor.items():
                groups.append({
                    "label": vendor,
                    "models": models,
                    "provider": "custom",
                    "base_url": base_url,
                    "api_key": api_key,
                    "is_library": True,
                })
    except Exception:
        pass

    return {
        "groups": groups,
        "minimum_context": _get_min_context(),
    }


class RemoteOllamaRequest(BaseModel):
    host: str
    token: str = ""


class RemoteOllamaPullRequest(BaseModel):
    host: str
    token: str = ""
    name: str


@app.post("/api/remote-ollama/models")
async def remote_ollama_models(req: RemoteOllamaRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: _fetch_remote_ollama_models(req.host, req.token))
    return JSONResponse(result)


@app.post("/api/remote-ollama/pull/stream")
async def remote_ollama_pull_stream(req: RemoteOllamaPullRequest):
    """Stream remote Ollama pull progress as SSE."""
    async def generate():
        import threading
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                base = req.host.rstrip("/")
                if not base.startswith("http"):
                    base = "http://" + base
                body = json.dumps({"name": req.name, "stream": True}).encode()
                headers = {"Content-Type": "application/json"}
                if req.token:
                    headers["Authorization"] = f"Bearer {req.token}"
                request = urllib.request.Request(
                    f"{base}/api/pull", data=body, headers=headers, method="POST",
                )
                with urllib.request.urlopen(request, timeout=1800) as resp:
                    for line in resp:
                        line = line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        q.put(("progress", obj))
                        if obj.get("status") == "success" or obj.get("error"):
                            break
                q.put(("done", {"ok": True, "status": "success"}))
            except Exception as e:
                q.put(("error", {"message": str(e)}))
            finally:
                q.put(("__end__", None))

        threading.Thread(target=worker, daemon=True).start()
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=0.25))
            except queue.Empty:
                yield ": ping\n\n"
                continue
            event, data = item
            if event == "__end__":
                break
            yield _sse(event, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Project CRUD ──────────────────────────────────────────────────────
class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""

class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class AddDocumentRequest(BaseModel):
    title: str
    content: str
    doc_type: str = "other"


@app.get("/api/projects")
async def list_projects():
    return JSONResponse(_project_store.list_projects())


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest):
    proj = _project_store.create_project(req.name, req.description)
    return JSONResponse(proj, status_code=201)


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    proj = _project_store.get_project(project_id)
    if not proj:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(proj)


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest):
    proj = _project_store.update_project(project_id, name=req.name, description=req.description)
    if not proj:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(proj)


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    ok = _project_store.delete_project(project_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ── Document CRUD ─────────────────────────────────────────────────────
@app.get("/api/projects/{project_id}/documents")
async def list_documents(project_id: str):
    return JSONResponse(_project_store.list_documents(project_id))


@app.post("/api/projects/{project_id}/documents")
async def add_document(project_id: str, req: AddDocumentRequest):
    proj = _project_store.get_project(project_id)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    doc = _project_store.add_document(project_id, req.title, req.content, req.doc_type)
    return JSONResponse(doc, status_code=201)


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = _project_store.get_document(doc_id)
    if not doc:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(doc)


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    ok = _project_store.delete_document(doc_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ── Context preview (for debugging) ──────────────────────────────────
@app.post("/api/projects/{project_id}/context")
async def preview_context(project_id: str, req: Request):
    body = await req.json()
    query = body.get("query", "")
    context = _project_store.build_project_context(project_id, query)
    return JSONResponse({"context": context, "token_estimate": len(context.split())})


# ── Session reset ─────────────────────────────────────────────────────
@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"ok": True}


# ── Serve frontend ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# Mount static files (CSS, JS, etc.) if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Hermes Web  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
