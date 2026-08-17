# GLOSSARY.md

> Every term teaching mode explains gets appended here, plainly, as it comes up. Shared across both
> workstreams — no split needed, explanations don't collide. Grows through the project; by Phase 8 it
> should make the README much easier to write.

**RAG (Retrieval-Augmented Generation).** Instead of an LLM answering purely from what it memorised
during training, you first search a document collection for relevant passages, then hand those
passages to the LLM as context and ask it to answer *using* them. Grounds the answer in real,
citable text instead of the model's (possibly wrong) memory.

**Chunking.** Splitting long documents into smaller pieces before indexing, because embedding
models and retrieval work best on paragraph-sized text, not whole documents. *How* you split
(fixed-size windows vs. sentence boundaries vs. the document's own structure) measurably changes
what gets retrieved later — hence six different strategies being compared rather than one guess.

**Embedding.** Converting text into a vector (list of numbers) such that semantically similar text
ends up as nearby vectors. This is what makes "search by meaning" possible instead of only "search
by exact keyword."

**Dense retrieval vs. sparse retrieval.** Dense = searching by embedding similarity (catches
meaning, e.g. "car" ≈ "automobile"). Sparse = keyword-based search like BM25 (catches exact terms,
e.g. a specific product code an embedding might blur past). Hybrid retrieval runs both and combines
the results, because each catches things the other misses.

**HNSW (Hierarchical Navigable Small World)** — the algorithm behind our dense vector index (FAISS
`IndexHNSWFlat`). Think of it as a multi-layer "skip list" for vectors: a sparse top layer lets you
jump close to the right neighborhood in a few hops, then denser lower layers refine the search. It
trades a small amount of recall for very fast approximate nearest-neighbor search — the reason a
200ms budget is even achievable with in-process search instead of a linear scan over millions of
vectors.

**RRF (Reciprocal Rank Fusion)** — how we combine a dense-search ranked list and a sparse (BM25)
ranked list into one ranking, without needing to normalize two incomparable score scales (cosine
similarity vs. BM25 score). Each document gets `1 / (k + rank)` from each list it appears in (k=60
here), using only rank position rather than raw score, and the contributions are summed. Simple,
robust, and — per the 2026 literature — the production default for exactly this kind of hybrid
retrieval.

**Deadline propagation** — every request enters the harness carrying a millisecond budget (200ms).
Each pipeline stage is told how much budget remains before it starts, and if it can't fit inside
that remainder, optional stages get skipped rather than the request blowing the deadline. It's the
difference between a system that degrades gracefully under load and one that just gets slower and
slower until it's unusable — analogous to a delivery driver with a hard cutoff time who skips a
low-priority stop rather than being late to everyone after it. This is the project's core
engineering bet.

**ONNX / int8 quantisation** — ONNX is a portable format for a trained model's computation graph, so
it can run outside the framework (PyTorch) that trained it, typically faster on CPU. int8
quantisation shrinks the model's numbers from 32-bit floats to 8-bit integers, trading a small
accuracy hit for a large speed and memory win — but only on CPU; on GPU it's actually *slower* than
FP32 because GPUs aren't optimized for int8 arithmetic the way CPUs increasingly are.

**E5 prefixes (`"query: "` / `"passage: "`)** — the `multilingual-e5-small` embedding model was
*trained* with these literal string prefixes prepended to its inputs, so it learned different
representations for "things being searched for" vs. "things being searched over." Omit the prefix at
inference time and the model still runs without error — it just silently returns worse embeddings,
because it's now off-distribution from its training data. This is the classic "no error, just wrong"
bug class, which is why there's a unit test asserting the prefix is applied rather than trusting code
review to catch it.

**BM25** — a classic lexical (keyword-matching) ranking function, the sparse counterpart to dense
vector search. It scores documents by term frequency, adjusted for document length and how rare a
term is across the whole corpus. Doesn't understand synonyms or semantics the way embeddings do, but
catches exact-term matches embeddings sometimes miss — which is why hybrid (dense + sparse via RRF)
beats either alone on most real corpora.

**Recall@k / MRR@10 / nDCG@10** — three ways of scoring a ranked retrieval result against ground
truth. Recall@k: was the right passage anywhere in the top k? MRR@10 (Mean Reciprocal Rank): if the
right passage was found, how high did it rank (1/rank, averaged over queries) — rewards ranking it
#1 much more than #8. nDCG@10 (normalized Discounted Cumulative Gain): like MRR but handles multiple
relevant passages per query with graded relevance, not just a single right answer.

**Parquet + row groups** — Parquet is a columnar file format (values of one field stored together,
not row-by-row) that compresses well and lets you read a subset of columns without touching the
rest. A large parquet file is internally chunked into "row groups" — each one independently
readable — which is what lets a reader stream through millions of rows without loading the whole
file into memory at once. This project reads `ai4bharat/MSMARCO-XI`'s Hindi file this way: a few
hundred rows for inspection cost a small memory footprint, not the whole 3.7GB.

**Hierarchical / small-to-big chunking** — index small, precise chunks (better for matching a
specific query tightly) but hand a larger surrounding "parent" chunk to the generation step (better
for having enough context to write a coherent answer). The two concerns — precision at retrieval
time, context at generation time — usually want different chunk sizes, so this strategy just gives
each its own size instead of compromising on one.

**Semantic chunking via similarity troughs** — instead of cutting text every N words, embed each
sentence and look at how similar consecutive sentences are to each other. Where similarity drops
sharply (a "trough"), the topic likely changed — that's where you cut. It costs more upfront (every
sentence needs an embedding, not just every chunk), but chunk boundaries end up aligned with actual
meaning shifts instead of an arbitrary word count.

**Groundedness / hallucination check.** Verifying that a generated answer's claims are actually
supported by the retrieved passages, rather than trusting the LLM not to make things up. The cheap
version (hot path) just checks that cited passage IDs were really retrieved and that the answer's
wording overlaps with the cited text; the expensive version (offline only) uses a model trained
specifically to judge factual entailment.

**Track A / Track B (extractive vs. generative answer).** Track A picks the single best-matching
span of text straight out of the top retrieved passage — fast (~15-30ms) and always available, but
not fluently phrased. Track B asks an LLM to synthesise a nicer-sounding answer, which takes longer
because it depends on a network call to an inference provider. Track A is shown first and Track B's
streamed answer replaces it if/when it arrives — the user is never left staring at a spinner, and
the 200ms budget is met honestly by the fast path even when the LLM is slow.
