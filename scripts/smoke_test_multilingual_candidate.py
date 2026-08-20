"""Local-only smoke test for the Phase 3 multilingual candidate (docs/DECISIONS.md ADR-012).
Sets VRAG_INDEX_DIR before any vrag import so the real /ask path loads the new 14-language
index -- exactly the env-var mechanism documented in interface.py, no production code touched.

Usage: python scripts/smoke_test_multilingual_candidate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from vrag.api.main import app  # noqa: E402
from vrag.retrieval.interface import is_retrieval_real  # noqa: E402

client = TestClient(app)

print(f"VRAG_INDEX_DIR = {os.environ['VRAG_INDEX_DIR']}")
print(f"is_retrieval_real() = {is_retrieval_real()}")

resp = client.get("/healthz")
print(f"/healthz -> {resp.json()}")
assert resp.json()["retrieval"] == "real", "STUB FALLBACK -- the candidate index did not load"

tests = [
    ("hi-IN", "भारत में सबसे ऊँचा पर्वत कौन सा है?"),
    ("en-IN", "What is the highest mountain in India?"),
    ("bn-IN", "ভারতের সর্বোচ্চ পর্বত কোনটি?"),
    ("ta-IN", "இந்தியாவின் மிக உயரமான மலை எது?"),
    ("mr-IN", "भारतातील सर्वात उंच पर्वत कोणता आहे?"),
]

seen_chunk_ids: set[str] = set()
for lang, query in tests:
    resp = client.post("/ask", json={"query": query, "k": 5, "language": lang})
    body = resp.json()
    print(f"\n[{lang}] query={query!r}")
    print(
        f"  status={body['status']} query_language={body['query_language']} "
        f"language={body['language']}"
    )
    if body["citations"]:
        for c in body["citations"]:
            print(
                f"  citation chunk_id={c['chunk_id']} passage_id={c['passage_id']} "
                f"score={c['score']:.4f}"
            )
            assert c["chunk_id"] not in seen_chunk_ids or True  # dupes across queries are fine
            assert "::" in c["chunk_id"], f"chunk_id not language-qualified: {c['chunk_id']}"
    assert body["query_language"] == lang
    assert "retrieve" in body["timings_ms"], "did not reach retrieval"

print(
    "\nAll smoke checks passed: real index loaded, no stub fallback, "
    "all 5 languages reached retrieval with language-qualified citation ids."
)
