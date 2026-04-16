"""
shannon-web — A minimal web frontend for Hermes Agent.

Run:
    cd shannon-web
    python server.py          # defaults to port 8080
    python server.py --port 9000
"""
import asyncio
import json
import queue
import re
import sys
import threading
import time
import traceback
import urllib.request
import uuid
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

# ── Load Hermes config once at startup ───────────────────────────────
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
# history is preserved across messages.
_sessions: dict[str, AIAgent] = {}
_sessions_lock = threading.Lock()

STATIC_DIR = Path(__file__).parent / "static"


def _get_or_create_agent(session_id: str, *, model: str = "",
                         provider: str = "", base_url: str = "",
                         api_key: str = "") -> AIAgent:
    with _sessions_lock:
        if session_id not in _sessions:
            d = _resolved_defaults()
            eff_model    = model    or d["model"]
            eff_provider = provider or d["provider"] or None
            eff_base_url = base_url or d["base_url"] or None

            # For custom/local providers (Ollama, vLLM, LM Studio) we need a
            # dummy API key so AIAgent takes the explicit-credentials path.
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
        else:
            # Session already exists — hotswap if a different model is requested
            agent = _sessions[session_id]
            if model and model != agent.model:
                agent.model = model
                if provider:
                    agent.provider = provider.strip().lower()
                if base_url:
                    agent.base_url = base_url
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


class SwitchModelRequest(BaseModel):
    model: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


# ── SSE event helpers ─────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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

    # Attach callbacks to the agent instance before this turn
    agent.stream_delta_callback = on_delta
    agent.tool_start_callback = on_tool_start
    agent.tool_complete_callback = on_tool_complete
    agent.reasoning_callback = on_reasoning
    agent.thinking_callback = on_thinking
    agent.tool_gen_callback = on_tool_gen

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

    def run():
        try:
            agent.run_conversation(
                effective_message,
            )
        except Exception as e:
            q.put(("error", str(e)))
            traceback.print_exc()
        finally:
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


def _fetch_cloud_provider_models(provider_id: str) -> list[dict]:
    """Fetch models for a specific cloud provider."""
    try:
        from hermes_cli.models import curated_models_for_provider, fetch_github_model_catalog

        if provider_id == "copilot":
            try:
                from hermes_cli.copilot_auth import resolve_copilot_token
                token, _ = resolve_copilot_token()
                if token:
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

    # If user supplied an API key, temporarily set it in env for this request
    user_key = body.get("api_key", "")
    import os
    old_val = os.environ.get(prov["env"])
    if user_key:
        os.environ[prov["env"]] = user_key

    try:
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, lambda: _fetch_cloud_provider_models(provider_id))
        return JSONResponse({
            "provider": provider_id,
            "base_url": prov["base_url"],
            "models": models,
        })
    finally:
        # Restore original env
        if user_key:
            if old_val is not None:
                os.environ[prov["env"]] = old_val
            else:
                os.environ.pop(prov["env"], None)


# ── Ollama pull (download a library model) ────────────────────────────
class OllamaPullRequest(BaseModel):
    name: str


def _ollama_pull_sync(name: str) -> dict:
    """Pull a model from Ollama registry (blocking)."""
    try:
        body = json.dumps({"name": name, "stream": False}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/pull",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
        return {"ok": True, "status": data.get("status", "success")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/ollama/pull")
async def ollama_pull(req: OllamaPullRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: _ollama_pull_sync(req.name))
    return JSONResponse(result)


# ── Hotswap model mid-session ─────────────────────────────────────────
@app.patch("/api/session/{session_id}/model")
async def switch_model(session_id: str, req: SwitchModelRequest):
    """Switch model on an existing agent, preserving conversation history."""
    with _sessions_lock:
        if session_id not in _sessions:
            return JSONResponse({"error": "session not found"}, status_code=404)
        agent = _sessions[session_id]

    # Update model name
    agent.model = req.model

    # Update provider / base_url if supplied
    new_provider = (req.provider or "").strip().lower()
    new_base_url = req.base_url or ""
    new_api_key = (req.api_key or "").strip()

    if new_provider:
        agent.provider = new_provider
    if new_base_url:
        agent.base_url = new_base_url

    # If the base_url changed, rebuild the underlying OpenAI client so it
    # actually hits the new endpoint.
    if new_base_url and new_base_url != str(getattr(agent, '_client_kwargs', {}).get('base_url', '')):
        from openai import OpenAI
        api_key = new_api_key or ("local" if new_provider == "custom" else getattr(agent.client, 'api_key', '') if agent.client else "")
        agent.client = OpenAI(api_key=api_key, base_url=new_base_url)
        agent._client_kwargs = {"api_key": api_key, "base_url": new_base_url}

    # Recalculate api_mode (mirrors AIAgent.__init__ logic)
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

    # Promote to Responses API when the model or direct-OpenAI URL requires it
    if (
        agent.api_mode == "chat_completions"
        and agent.provider != "copilot-acp"
        and not str(agent.base_url).lower().startswith("acp://copilot")
        and not str(agent.base_url).lower().startswith("acp+tcp://")
        and (agent._is_direct_openai_url() or agent._model_requires_responses_api(agent.model))
    ):
        agent.api_mode = "codex_responses"

    # Recalculate prompt-caching eligibility
    is_openrouter = agent._is_openrouter_url()
    is_claude = "claude" in agent.model.lower()
    is_native_anthropic = agent.api_mode == "anthropic_messages" and agent.provider == "anthropic"
    agent._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic

    return {"ok": True, "model": agent.model, "provider": agent.provider, "api_mode": agent.api_mode}


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


def _remote_ollama_pull_sync(host: str, token: str, name: str) -> dict:
    """Pull a model on a remote Ollama instance (blocking)."""
    try:
        data = _remote_ollama_request(host, "/api/pull", token,
                                      method="POST",
                                      body={"name": name, "stream": False},
                                      timeout=600)
        return {"ok": True, "status": data.get("status", "success")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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


@app.post("/api/remote-ollama/pull")
async def remote_ollama_pull_endpoint(req: RemoteOllamaPullRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: _remote_ollama_pull_sync(req.host, req.token, req.name))
    return JSONResponse(result)


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
