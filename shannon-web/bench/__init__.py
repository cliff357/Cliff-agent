"""Benchmark subsystem for shannon-web.

Lets users:
  - Run task suites against different patterns (single agent / dual-agent review / ...)
  - Grade with LLM-as-judge or keyword/regex matching
  - Save historical reports in SQLite
  - Compare runs side-by-side
"""
from .runner import BenchRunner
from .store import BenchStore
from .suites_loader import load_suite, list_suites

__all__ = ["BenchRunner", "BenchStore", "load_suite", "list_suites"]
