# GLOSSARY.md

Grows every session teaching mode explains a term. Shared — either workstream appends, no split
needed.

- **RAG (Retrieval-Augmented Generation).** Instead of an LLM answering purely from what it
  memorised during training, you first search a document collection for relevant passages, then
  hand those passages to the LLM as context and ask it to answer *using* them. Grounds the answer
  in real, citable text instead of the model's (possibly wrong) memory.

- **Chunking.** Splitting long documents into smaller pieces before indexing, because embedding
  models and retrieval work best on paragraph-sized text, not whole documents. *How* you split
  (fixed-size windows vs. sentence boundaries vs. the document's own structure) measurably changes
  what gets retrieved later — hence six different strategies being compared rather than one guess.

- **Embedding.** Converting text into a vector (list of numbers) such that semantically similar
  text ends up as nearby vectors. This is what makes "search by meaning" possible instead of only
  "search by exact keyword."

- **Dense retrieval vs. sparse retrieval.** Dense = searching by embedding similarity (catches
  meaning, e.g. "car" ≈ "automobile"). Sparse = keyword-based search like BM25 (catches exact
  terms, e.g. a specific product code an embedding might blur past). Hybrid retrieval runs both
  and combines the results, because each catches things the other misses.

- **HNSW (Hierarchical Navigable Small World).** The dense-index algorithm FAISS uses. Think of it
  as a multi-layer shortcut graph over your vectors: instead of comparing a query against every
  single stored vector (slow), it hops through a small number of "highway" nodes first, then
  narrows in — giving near-exact search results in milliseconds even over millions of vectors.

- **RRF (Reciprocal Rank Fusion).** A way to combine two ranked lists (dense results + sparse
  results) into one, using only each item's *rank position* in each list rather than its raw
  score. This sidesteps the problem that a cosine-similarity score and a BM25 score live on
  completely different numeric scales and can't be averaged directly.

- **Deadline propagation.** Every request carries a millisecond budget. As it moves through the
  pipeline, each stage checks how much time is left; if an optional stage (like reranking) can't
  fit, it's skipped rather than run anyway and blowing the deadline. The system degrades quality
  under pressure instead of degrading latency — it can never take longer than promised, only get
  slightly less thorough. This is the project's core engineering bet.

- **Groundedness / hallucination check.** Verifying that a generated answer's claims are actually
  supported by the retrieved passages, rather than trusting the LLM not to make things up. The
  cheap version (hot path) just checks that cited passage IDs were really retrieved and that the
  answer's wording overlaps with the cited text; the expensive version (offline only) uses a
  model trained specifically to judge factual entailment.

- **Track A / Track B (extractive vs. generative answer).** Track A picks the single best-matching
  span of text straight out of the top retrieved passage — fast (~15-30ms) and always available,
  but not fluently phrased. Track B asks an LLM to synthesise a nicer-sounding answer, which takes
  longer because it depends on a network call to an inference provider. Track A is shown first and
  Track B's streamed answer replaces it if/when it arrives — the user is never left staring at a
  spinner, and the 200ms budget is met honestly by the fast path even when the LLM is slow.
