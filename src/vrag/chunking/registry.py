"""Self-registration for chunking strategies. Strategy modules call `register()` at import time;
scripts/eval_chunking.py imports `vrag.chunking.strategies` (which imports every strategy module)
and then enumerates via `all_strategies()` — no per-strategy code change needed to add one to the
eval matrix.
"""

from __future__ import annotations

from .base import ChunkingStrategy

_REGISTRY: dict[str, ChunkingStrategy] = {}


def register(strategy: ChunkingStrategy) -> ChunkingStrategy:
    if strategy.name in _REGISTRY:
        raise ValueError(f"Chunking strategy '{strategy.name}' already registered")
    _REGISTRY[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> ChunkingStrategy:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown chunking strategy '{name}'. Registered: {sorted(_REGISTRY)}"
        ) from None


def all_strategies() -> dict[str, ChunkingStrategy]:
    return dict(_REGISTRY)
