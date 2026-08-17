"""Individual ChunkingStrategy implementations, one module per strategy. Owner: Workstream R.

Importing this package registers every strategy (each submodule calls `registry.register()` at
import time) — this is the only place that needs editing when a new strategy is added; nothing
else in eval_chunking.py or the registry needs to change.
"""

from vrag.chunking.strategies import (  # noqa: F401
    fixed_overlap,
    hierarchical,
    metadata_aware,
    passage_native,
    semantic,
    sentence_window,
)

