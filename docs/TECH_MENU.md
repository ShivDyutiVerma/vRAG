# TECH_MENU.md
## Candidate methods per pipeline stage + the experiment plan

> Lives at `docs/TECH_MENU.md`. Companion to `AGENT_BUILD_SPEC.md` (architecture) and
> `docs/BUILD_PLAN.md` (execution).
>
> **Verdict legend:**
> `SHIP` = default, use it, don't debate · `TEST` = genuine candidate, must be measured in the ablation
> `BENCH-ONLY` = too slow for the hot path, but use it in offline eval to prove rigor · `REJECT` = ruled out, reason given

---

# §A. Why you must NOT try every combination

The full space is roughly:

```
6 chunking × 4 embedding × 3 retrieval mode × 3 rerank × 3 generation = 648 configurations
```

Each chunking or embedding change forces a **full re-index** (minutes to an hour). 648 configs is
physically impossible in 5 days between two people, and worse, it's bad methodology: with that many trials on a 500-query
eval set you will pick a winner that's inside the noise band and won't reproduce.

**Use staged greedy ablation instead: 19 runs, ~95% of the value.**

Optimise one axis at a time, in dependency order, freezing everything upstream:

| Stage | Vary | Hold fixed | Runs |
|-------|------|-----------|------|
| A1 | Chunking (6) | embedder = e5-small, dense-only, no rerank | 6 |
| A2 | Embedder (4) | chunking = A1 winner | 4 |
| A3 | Retrieval mode (3): dense / sparse / hybrid+RRF | A1+A2 winners | 3 |
| A4 | Rerank (3): none / FlashRank / cross-encoder | A1–A3 winners | 3 |
| A5 | Generation (3) | A1–A4 winners | 3 |
| **Total** | | | **19** |

**Then run one confirmation pass:** take the top-2 from each of A1 and A2 and test the 4 cross
combinations. Interaction effects between chunking and embedder are the only ones that genuinely
matter here (late chunking, for instance, only works with long-context token-level models). That's
23 runs total — achievable in one day if the eval harness is automated.

### Rules that make the ablation trustworthy
1. **Freeze the eval set before run 1.** 500 held-out query→passage pairs, committed to git, never regenerated.
2. **One variable per run.** If two things changed, the run is void.
3. **Every run appends a row to `eval/ablation_ledger.csv`** — config hash, all params, all metrics, wall-clock, timestamp, git SHA.
4. **Report the noise floor.** Run the *same* config 3× and report the spread. Any "winner" inside that spread is not a winner — say so.
5. **Latency and quality are measured in the same run.** A config that wins Recall@5 by 1pt and costs 40ms loses.

---

# §B. Stage-by-stage menu

## S1 — Audio capture & VAD (client + server edge)

| Option | Notes | Verdict |
|--------|-------|---------|
| Browser `MediaRecorder` → WebSocket PCM | Simplest path. Sarvam streaming wants WAV or raw PCM (`pcm_s16le`/`pcm_l16`/`pcm_raw`); PCM is 16kHz only. Other formats are not supported for real-time streaming. | **SHIP** |
| Server-side VAD (Sarvam's built-in) | Sarvam's realtime endpoint runs its own VAD and emits speech-start/speech-end events, tunable via threshold and silence duration. Less code for you. | **SHIP** |
| Client-side VAD (Silero WASM) | Cuts bandwidth, adds complexity. Only if bandwidth is a real problem. | **TEST** (P7 stretch) |
| `webkitSpeechRecognition` | Violates constraint C1. | **REJECT** |

**Gotcha:** browsers refuse `getUserMedia` on non-HTTPS network origins. Confirm HTTPS in Phase 1, not Phase 8.

---

## S2 — Speech-to-text (constraint C1: Sarvam or ElevenLabs, pick one)

| Option | Latency | Language fit | Verdict |
|--------|---------|-------------|---------|
| **Sarvam `saaras:v3-realtime`** (WebSocket) | True interim transcripts, millisecond VAD tuning, live mid-call reconfiguration; `stream_type` trades partial-latency vs accuracy (`fast`/`balanced`/`simulated`) | 22–24 Indic languages, code-mixing, India-hosted | **SHIP** |
| Sarvam legacy streaming (`/speech-to-text/ws`, saaras:v3) | Generally available, superseded for voice-agent work | same | **TEST** (fallback if realtime is beta-flaky) |
| Sarvam REST (batch) | Simplest to build | same | **SHIP for Phase 1 only**, then migrate to WS |
| ElevenLabs Scribe v2 Realtime | ~150ms, 90+ languages, no diarization in realtime mode | Broad, but strongest in English/major European/East Asian | **REJECT** per ADR-001 (corpus is Indic) |

**Modes to know:** Saaras v3 exposes `transcribe` (default), `translate`, `verbatim`, `translit`, `codemix`.
`translit` (Latin script) and `codemix` are worth testing — if your index is Devanagari, `transcribe` keeps
query and corpus in the same script, which matters for BM25 matching. **Test `transcribe` vs `translit` for
retrieval quality in A3.** This is a genuinely novel axis most teams won't think of.

---

## S3 — Input guardrails (hot path, must be <5ms)

| Option | Latency | Verdict |
|--------|---------|---------|
| Regex/keyword denylist + length/degeneracy checks | ~1-2ms, in-process | **SHIP** |
| Llama Prompt Guard 2 (86M) | 20–50ms on H100 with FP8 and short inputs — on CPU, far more | **BENCH-ONLY** |
| Llama Guard 3/4 (8B) | Hazard-category classification; way outside budget | **BENCH-ONLY** |
| NeMo Guardrails (Colang DSL) | Six rail types incl. a retrieval rail; latency scales with flow complexity, multi-step flows compound | **REJECT** for hot path — heavy, and a DSL to learn with 5 days left |
| Guardrails AI validator hub | Composition without writing orchestration; validator quality is uneven, audit before adopting | **TEST** only if you want an off-the-shelf PII validator |
| Presidio (PII) | Mature PII detection/redaction for G5 | **TEST** |

**Design principle the guardrail literature is unanimous on: cheap regex first, classifiers second,
LLM judges only on ambiguous cases.** Never spend ML budget on patterns regex catches in 2ms.
A guardrail sitting at 400ms p50 becomes the latency story for the whole product.

---

## S4 — Query embedding (hot path — this is where the biggest win is available)

| Option | Dim | Query latency (CPU) | Indic coverage | Verdict |
|--------|-----|--------------------|----------------|---------|
| **`potion-multilingual-128M`** (Model2Vec static) | 256 | **sub-ms** — static lookup + mean pool, up to 500× faster than the source transformer | 101 languages | **TEST — potentially the whole ballgame** |
| **`multilingual-e5-small` + ONNX int8** | 384 | ~2.5ms measured on Graviton3-class CPU | 100 langs | **SHIP as default** |
| `multilingual-e5-base` + ONNX int8 | 768 | ~2-3× the small | 100 langs | **TEST** |
| **BGE-M3** | 1024 | Heaviest of the four | 100+ langs, **and it's the only one that emits dense + sparse + multi-vector from one model** | **TEST** (see note) |
| `jina-embeddings-v3` | 1024 (MRL→32) | 570M params | Strong multilingual, 8192 ctx | **TEST** — required if you want late chunking |
| `Vyakyarth-1-Indic` (Krutrim) | 768 | XLM-R based | 10 Indic langs + English, contrastive-trained, benchmarked on FLORES Indic retrieval | **TEST — the Indic-specialist wildcard** |
| MuRIL / IndicBERT | — | Not contrastively trained for retrieval | — | **REJECT** — these are NLU encoders, not sentence-retrieval models. Common beginner mistake. |

**Three things that will bite you:**
- **E5 needs prefixes.** `"query: "` and `"passage: "`. Omit them and recall silently collapses. Write the test first.
- **Static embeddings lack contextualisation** — that's the quality tradeoff. Potion-class models sit in a different quality band from transformers. Measure it, don't assume it's free.
- **Matryoshka truncation** (jina-v3, EmbeddingGemma) lets you cut dimensions to shrink the index and speed search — jina-v3 retains ~92% of retrieval performance at 64 dims vs 1024. A cheap latency lever worth one ablation run.

**BGE-M3 special case:** it does dense, sparse, and ColBERT-style multi-vector retrieval in a single
model. If it wins A2, your "hybrid retrieval" story becomes unusually elegant — one model, three
retrieval modes, fused. Worth testing for the narrative alone.

---

## S5 — Chunking (offline, so quality is nearly free — this is requirement C2)

| # | Strategy | Evidence | Verdict |
|---|----------|----------|---------|
| 1 | **Fixed-size / recursive character + overlap** | The strongest general default; ~512 tokens is the standard starting point (256 for short Q&A). 10–20% overlap is the conventional range. | **SHIP as baseline** |
| 2 | **Passage-native** (dataset boundaries) | MS MARCO passages are already coherent retrieval units | **TEST — likely winner on this corpus** |
| 3 | **Sentence-window** | Retrieve on a sentence, return sentence±N for generation. Decouples retrieval granularity from generation granularity. Matches semantic quality up to ~5k tokens for far less cost. | **TEST** |
| 4 | **Semantic (embedding-similarity boundaries)** | Can improve recall up to ~9%, but Chonkie benchmarks put it ~14× slower than token-based (≈0.33 MB/s vs 4.82 MB/s) | **TEST** — offline cost only, so acceptable |
| 5 | **Metadata-aware** (language/source_lang/query-type tags) | Most dataset-specific; enables language-filtered or boosted retrieval | **TEST — highlight this one** |
| 6 | **Hierarchical / parent-child (small-to-big)** | Small chunks for precision, parent passage for generation context | **TEST** |
| 7 | **Late chunking** | Embed the whole document first, then split token embeddings — chunks carry document-level context. Cheaper than contextual retrieval (embedding model only, no LLM pass). **Requires a long-context token-level model (jina-v3, BGE-M3).** With jina-v3 + late chunking, fixed and semantic chunking barely differ — the document context already carries the cross-references. | **TEST — the "we read the 2026 literature" flex** |
| 8 | **Contextual Retrieval** (Anthropic-style: LLM prepends context to each chunk) | Same problem as late chunking, solved with an LLM pass. Best paired with BM25 + reranking. Expensive to index. | **TEST only if time allows** — it's an LLM call per chunk |
| 9 | **Proposition / atomic-fact** | LLM decomposes passages into self-contained facts | **STRETCH** |

**Two findings worth putting in your README:**
- A January 2026 study found **overlap provides no measurable benefit with SPLADE (sparse) retrieval** — so the "always use 10-20% overlap" rule is not universal. Test overlap as its own variable rather than assuming it.
- Chunking choice can swing recall by up to ~9% on the same corpus. That's the size of the prize, and it justifies the whole ablation.

**Optimise for faithfulness, not just retrieval recall.** The chunker that maximises Recall@5 is not
automatically the one that produces the best final answers. Report both.

---

## S6 — Dense index (hot path)

| Option | Evidence | Verdict |
|--------|----------|---------|
| **FAISS `HNSW32`, inner product on normalised vectors** | Industry standard; benchmarked faster than ScaNN and on par with Vamana on Deep10M | **SHIP** |
| `hnswlib` | Head-to-head at small scale: 2,194 QPS @ 0.995 recall on a 45K×1024 set; 5,184 QPS @ 0.827 on glove-100 — competitive with or faster than usearch f32 | **TEST in P6 if embedding+search shows up as a bottleneck** |
| `usearch` (int8 quantised) | 5,726 QPS @ 0.928 recall on the same 45K set at 55MB vs 191–202MB for f32 — big memory win for a modest recall cost | **TEST** — matters if host RAM is tight |
| FAISS IVF-PQ | 2,597 QPS @ 0.936, 40MB — smallest memory | **TEST** if RAM-constrained |
| Qdrant / Pinecone / Weaviate (hosted) | **50–300ms network round trip.** Eats the entire budget. | **REJECT** — this is the single most important "don't" in the project |
| `sqlite-vec` brute force | 27 QPS at 45K rows; impractical at 1.18M | **REJECT** |

**Tune `efSearch`, don't guess it.** Produce a recall-vs-latency curve and pick the operating point
from the data — this is a Phase 3 deliverable and a great chart for the README.

---

## S7 — Sparse / lexical index (hot path)

| Option | Evidence | Verdict |
|--------|----------|---------|
| **`bm25s`** | Eager sparse scoring in scipy; up to 500× faster than rank-bm25, comparable to or exceeding Elasticsearch. A production RAG paper measured 103 queries in **0.05s vs 17s for BM25Okapi with essentially identical accuracy (94.24 vs 94.49)**. | **SHIP — not a real decision** |
| `rank_bm25` | The 17s side of that comparison | **REJECT** |
| SPLADE (learned sparse) | Better than BM25 on recall, but needs a neural forward pass at query time | **TEST** only if S4 lands sub-ms |
| BGE-M3 sparse mode | Free if BGE-M3 wins A2 | **TEST** (bundled) |
| Elasticsearch / OpenSearch | Java, network hop, ops burden | **REJECT** |

**Indic tokenisation is the real risk here.** `bm25s` ships a fast tokenizer built around English
stemming and stopwords. For Devanagari you need Unicode word-boundary tokenisation and no English
stemmer. Naive `.split()` will appear to work and quietly halve your sparse recall. **Write a unit
test that asserts a Hindi sentence tokenises into the expected token count.** Record the limitation
in `DECISIONS.md`.

---

## S8 — Fusion

| Option | Verdict |
|--------|---------|
| **Reciprocal Rank Fusion (RRF), k=60** | **SHIP** — no score normalisation needed across heterogeneous scorers, near-zero cost |
| Weighted score fusion (α·dense + (1−α)·sparse) | **TEST** — requires normalising incomparable score scales; sometimes beats RRF with tuning |
| Distribution-based normalisation | **STRETCH** |

`BM25 + dense + RRF + rerank` is repeatedly described as the 2026 production default. You're not
being exotic; you're being correct.

---

## S9 — Reranking (hot path, budget-gated)

| Option | Measured latency | Verdict |
|--------|-----------------|---------|
| **None** | 0ms | **SHIP as default** — prove rerank earns its ms |
| **FlashRank** (ONNX, quantised, CPU) | **sub-20ms for 50 candidates on CPU**; often the only practical option on CPU-only infra | **TEST — the only viable hot-path reranker for you** |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | ~1,800 docs/sec; nDCG@10 74.30 on TREC DL19 | **TEST** (ONNX it first) |
| `bge-reranker-v2-m3` | **241.0ms avg / 239.4ms P50 / 271.4ms P95** — exceeds your entire budget alone | **BENCH-ONLY** |
| Cohere Rerank 3.5 | ~220ms P50 + network | **REJECT** |
| ColBERT / PLAID late interaction | Precompute doc vectors, cheap MaxSim at query time; storage ~5-10× a dense index after ColBERTv2 residual compression | **TEST** if you want a genuinely differentiated approach |
| LLM/listwise rerankers (RankGPT, jina-reranker-v3) | Best quality, worst latency | **BENCH-ONLY** |

**Use the `rerankers` library** (`AnswerDotAI/rerankers`) — one unified API across cross-encoders,
FlashRank, ColBERT, and API rerankers. It turns A4 from four integrations into four config strings.
That single choice probably saves you half a day.

---

## S10 — Generation (the budget crunch)

| Option | TTFT evidence | Verdict |
|--------|--------------|---------|
| **Groq** | ~120ms median TTFT, 300+ TPS; P95 200–400ms (described as "marginal" for voice, needing prompt optimisation) | **TEST — probe RTT from India first** |
| **Sarvam LLM** | India-hosted → lowest network RTT from an Indian deployment | **TEST — probe** |
| SambaNova | ~150ms median TTFT | **TEST** |
| Fireworks | ~180ms median; P95 350–700ms | **TEST** |
| **Local small model** (Qwen2.5-1.5B via llama.cpp) | Zero network RTT; needs CPU/RAM headroom | **TEST — the guaranteed-budget escape hatch** |
| OpenAI / Anthropic / Gemini / Bedrock | Gemini ~600ms median TTFT; the low-latency literature explicitly rules these out as primary providers for voice AI at current performance | **REJECT** for the hot path |

**Non-negotiable technique regardless of provider:** stream, and **begin emitting on the first
sentence** rather than waiting for the full response. Streaming reduces *perceived* latency by ~75%.
This is exactly what your Track A/Track B split operationalises.

**Watch tail latency, not just median.** One provider benchmarked at 300ms median but 2,500ms P95 —
a "fastest tail-latency disaster." Since you must report **P100**, a provider with a bad tail will
wreck your headline number even if its median looks great. Probe P95/P100, not averages.

---

## S11 — Structured output (requirement C5)

| Option | Overhead | Verdict |
|--------|----------|---------|
| Provider-native JSON schema mode | Zero integration cost | **SHIP if your chosen provider supports it** |
| **XGrammar** | **<40µs per token**; default backend in vLLM, SGLang, TensorRT-LLM; up to 3× on JSON Schema and >100× on CFG vs best baseline; schema compile 0.12–0.30s | **SHIP for local/vLLM** |
| **llguidance** (Microsoft, Rust Earley parser) | ~50µs/token, negligible startup; faster TTFT on dynamic schemas; benchmarked at 6–9ms/token vs 15–16ms unconstrained | **TEST** |
| Outlines | Compilation of 40s to 10+ min on complex schemas; lowest compliance rate in JSONSchemaBench largely due to timeouts | **REJECT** |
| Pydantic validate + retry | Your fallback layer, not your primary | **SHIP as layer 2** |

**Two gotchas worth an entry in `CONVENTIONS.md`:**
- **Schema compilation is 50–200ms on first request**, then cached to near-zero. **Compile your schema at boot**, inside the warm-up, or your first user pays for it. This is exactly the kind of thing that silently ruins a demo.
- **Put the reasoning field BEFORE the answer field** in your schema. If `answer` comes first, the model commits before thinking. Also: keep nesting under 4 levels and always give fields descriptions.

Counter-intuitively, constrained decoding often *reduces* total latency — no conversational filler,
and generation stops the moment the JSON closes.

---

## S12 — Groundedness / hallucination check (requirement C6)

**Two tiers. This is the design that wins.**

**Hot path (<10ms, deterministic):**
| Check | Verdict |
|-------|---------|
| Citation-ID validation — every cited `chunk_id` must exist in what was actually retrieved | **SHIP** |
| N-gram / lexical overlap between answer and cited spans | **SHIP** |
| Answer-vs-context embedding similarity (you already have the vectors) | **TEST** |

**Offline eval (proves the hot-path check is a good approximation):**
| Model | Notes | Verdict |
|-------|-------|---------|
| **Bespoke-MiniCheck** | Tops LLM-AggreFact at 77.4%; ~200ms on a modern GPU, <100ms optimised; **works sentence-by-sentence and returns claim-level spans**; runs on MacBook-class hardware | **BENCH-ONLY — use it, it's the best fit** |
| Patronus Lynx (8B/70B) | Open source, PASS/FAIL with chain-of-thought | **BENCH-ONLY** |
| Vectara HHEM | Message-level score only | **TEST** |
| RAGAS `faithfulness` | Decomposes answers into claims and checks each against retrieved context; ~95% human agreement on faithfulness | **SHIP for the eval report** |
| DeepEval | Ships faithfulness + G-Eval with a CI test runner | **TEST** |

**The trap to avoid, stated bluntly in the literature:** an LLM judge will happily score an answer
"grounded" while the retrieval trace is empty, and invented citations pass judges surprisingly often
when nobody inspects sources. **Run deterministic retrieval checks BEFORE any groundedness judge.**
Your citation-ID validation is that deterministic check — it's not a cheap approximation, it's the
correct first gate.

---

## S13 — Caching

| Option | Verdict |
|--------|---------|
| Semantic cache (GPTCache-style) | **REJECT** — and cite why. The dual-agent voice-RAG paper that popularised this found local FAISS search takes ~0.1ms, making the cache pointless; its 316× speedup was against a *remote* Qdrant at 110ms. Your index is in-process. **Saying this out loud in your README proves you read the paper rather than cargo-culting it.** |
| Predictive prefetch / "Slow Thinker" background agent | **REJECT** for the same reason, but the fast/slow split maps onto your Track A/B design |
| Warm-up cache (models + schema compiled at boot) | **SHIP** — different thing entirely, and mandatory |
| Embedding cache for repeated benchmark queries | **REJECT during benchmarking** — it would falsify your latency numbers. Disable it in `bench_latency.py` and say so. |

---

## S14 — Harness / orchestration

| Option | Verdict |
|--------|---------|
| **Hand-rolled typed stages + Pydantic + `tenacity`** | **SHIP** — full control over the deadline propagation, which is your differentiator. A framework will fight you on it. |
| LangChain / LlamaIndex | **REJECT** — abstraction overhead you can't profile, and the graders want to see *your* orchestration |
| LangGraph | **TEST** only if the team already knows it |
| `asyncio.gather` for parallel dense∥sparse | **SHIP** |

---

## S15 — Telemetry & eval

| Option | Verdict |
|--------|---------|
| `time.perf_counter_ns()` + JSONL trace records | **SHIP** |
| OpenTelemetry spans | **TEST** — nice for the write-up, extra scaffolding |
| Arize Phoenix (open source) | **TEST** — trace-level RAG observability if you want dashboards |
| `polars`/`pandas` + `matplotlib` for percentiles and charts | **SHIP** |

---

## S16 — Deployment

| Option | Verdict |
|--------|---------|
| Single container, region chosen from the Phase 0 RTT probe, serving API + built frontend on one origin | **SHIP** |
| HF Spaces | **SHIP as deploy insurance** — stand it up in Phase 1 |
| Serverless (Lambda/Cloud Run min-instances=0) | **REJECT** — cold starts destroy the budget |

---

# §C. The measurement contract

Every ablation row in `eval/ablation_ledger.csv`:

```
run_id, timestamp, git_sha, config_hash,
chunk_strategy, chunk_params, embedder, embed_backend, embed_dim,
index_type, ef_search, retrieval_mode, fusion_k, reranker, top_k,
generator, prompt_version,
recall@1, recall@5, recall@10, mrr@10, ndcg@10,
faithfulness, abstention_rate_in_domain, abstention_rate_ood,
p50_ms, p70_ms, p95_ms, p100_ms,
p50_embed_ms, p50_search_ms, p50_rerank_ms, p50_ttft_ms,
index_build_s, index_size_mb, rss_mb,
notes
```

If a run doesn't produce a row, it didn't happen. If two rows differ in more than one config column,
neither is evidence.

---

# §D. What to say in the README

The judged artifact isn't the winning config — it's the **decision trail**. Structure it as:

1. Here is the space we considered (this document, condensed to a table)
2. Here is why we didn't grid-search it (§A — combinatorics + noise)
3. Here is the staged ablation and the ledger
4. Here is the noise floor, so you know which gaps are real
5. Here is what won, what we shipped, and the one place we chose the slower option on purpose
6. Here is what we'd test next with two more weeks

Point 5 matters. Every team will claim they picked the fastest thing. A team that says
"FlashRank won on quality but cost 18ms, so we made it budget-gated rather than default, and here's
the degradation test proving it sheds correctly under load" is doing engineering.
