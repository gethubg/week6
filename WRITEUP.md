# Trade Coach — Build Writeup

## What's built

All 7 milestones from the Product Spec, each landed as its own phase with a green pytest
suite before moving on:

| Phase | Contents | Tests |
|---|---|---|
| 1 | SQLite schema, FIFO lot matching, Ingestion Agent (Robinhood CSV → Trade/Leg) | 10 |
| 2 | Custom MCP server wrapping yfinance, Analysis Agent, TimingScore algorithm | 18 |
| 3 | Coach + Critic agents on LangGraph, retry/strip loop, LangSmith tracing hook | 12 |
| 4 | News-context + behavioral-concept-KB RAG (Pinecone-swappable vector store) | 12 |
| 5 | Pattern memory table + semantic merge across runs | 4 |
| 6 | Streamlit UI (Scorecard / Patterns / Errors tabs) wired to the full graph | 8 |
| 7 | Golden-dataset deterministic eval + Critic precision/recall + LLM-as-judge | 10 |

**90 tests, all offline.** Every phase runs and is verified without any API key, using a
small set of fakes that stand in for the real integrations:

- `llm.client.FakeLLMClient` — scripted structured-output responses instead of Anthropic.
- `mcp_server.price_provider.FakePriceProvider` — canned OHLCV bars instead of yfinance/network.
- `memory.vector_store.InMemoryVectorStore` + `memory.embeddings.HashEmbedder` — cosine
  similarity over a deterministic bag-of-words hash instead of Pinecone + OpenAI embeddings.

Every one of these has a real counterpart (`AnthropicLLMClient`, `YFinancePriceProvider`,
`PineconeVectorStore`, `OpenAIEmbedder`) already wired into `app/pipeline.py`, selected
automatically based on which keys are present in `.env`. Flipping from offline to live is
adding keys, not writing code.

## Architecture as actually wired

```
START → ingestion_node → analysis_node → retrieval_node → coach_node → critic_node
                                                                ↑            │
                                                     bump_retry_node ← "retry" (max 1)
                                                                             │
                                                          "finalize" → finalize_node → memory_node → END
```

`retrieval_node` (added while wiring Phase 6) is the piece connecting Phases 4–5 into the
live graph: it populates `news_context` (RAG, ±3 day window per trade leg), `patterns_retrieved`
(existing Pattern rows for this user), and `retrieved_concepts` (the full ~10-doc concept KB —
see note below) before the Coach ever runs. `memory_node` reconciles the Coach's approved
pattern claims into the Pattern table + patterns vector index after the Critic has had its say.

## One deliberate deviation from the tech spec: concept retrieval timing

Tech Spec §9.2 describes concept retrieval as "query the concepts index with the pattern
description as the query text" — but the Coach doesn't have a pattern description to query
with until *after* it runs. That's a genuine chicken-and-egg with a single-shot structured-output
LLM call (no mid-generation tool use). Given the concept KB is only ~10 short docs, the fix
was to hand the Coach the full list of concept titles up front (cheap at this scale) rather
than a similarity-gated shortlist. The threshold-based `rag/concepts.py::retrieve_concept`
function still exists and is tested — it's used by the eval suite (Phase 7) to double-check
that a Coach's citation was actually the closest KB match for its own claim description, not
just *a* valid title. A future version with real multi-turn tool use could move this back to
the Coach's own reasoning loop.

## Grounding: what the Critic actually catches

`tools/critic_verify.py` runs three checks with zero LLM involvement:
1. Every cited trade ID / score ID must exist in this run's data.
2. Every cited concept must be one of the concepts made available this run.
3. For patterns with a registered direction checker (currently: disposition effect),
   the evidence trades must actually support the claimed direction — e.g. a "sells winners
   early" claim requires the evidence trades' sell-leg percentiles to sit below the sell-side
   average, not above it.

The `evals/groundedness_eval.py` precision/recall harness (8 hand-labeled synthetic claims,
4 grounded / 4 not) currently scores 1.0/1.0 — it exists as a regression guard, so a future
change to the Critic's logic that silently breaks one of these cases fails loudly.

Separately, `judge_groundedness` is an LLM-as-judge over narrative *quality* (does the prose
actually match the numbers it cites?) — a check the deterministic Critic structurally can't do,
since it only verifies citations point to real data, not that the sentence describing them is
accurate.

## Live integration hardening (post-Phase-6)

The offline fake suite (`InMemoryVectorStore`, `FakePriceProvider`, etc.) proved the graph's
wiring, but running the real integrations end to end — real Finnhub, real Pinecone, real
Nebius embeddings, actual yfinance network egress — surfaced bugs the fakes structurally
couldn't catch, since they don't reproduce the real APIs' actual constraints:

- **News provider swap.** The original NewsAPI.org integration (`/v2/everything`) only
  returns articles from the last ~30 days on its free tier, so any trade leg older than a
  month got an empty news list rather than an error — a silent gap, not a crash.
  `mcp_server/news_provider.py` now wraps Finnhub's `/company-news` endpoint instead: ticker-
  native (not a keyword search that can match unrelated articles), and its free tier's actual
  lookback was empirically measured at ~12 months (Finnhub doesn't document an exact cutoff).
  Trade legs older than that still get no news — same failure class, longer window.
- **Pinecone rejects string range filters.** `retrieve_news_context`'s `$gte`/`$lte` date
  filter was built on the ISO date string, which sorts correctly in Python but Pinecone's real
  API rejects outright (`"$gte operator must be followed by a number"`). Fixed by adding a
  numeric `date_ordinal` (YYYYMMDD int) metadata field for range filtering, keeping the ISO
  string for display. `InMemoryVectorStore` never exercised this path since Python string
  comparison silently "worked" there.
- **News-to-leg join was exact-date, not windowed.** `retrieve_news_context` deliberately
  retrieves news in a ±3-day window around a leg's date (Tech Spec §9.1's whole point — catch
  "sold a day before earnings"), but `view_model.py` was joining news back to a leg by exact
  `(ticker, date)` equality. An article published even one day off a leg's date silently
  dropped off that leg's scorecard row. Fixed by having `retrieval_node` keep a
  `news_by_leg: dict[leg_id, list[NewsContext]]` — it already knows the per-leg association at
  retrieval time; the bug was throwing that away and trying to reconstruct it from a mismatched
  key.
- **Unbounded per-article network calls.** `fetch_and_index_news` indexed every article a
  provider returned, one `embed()` + one `upsert()` network round-trip each. Finnhub returns
  200+ articles for a single week for a mega-cap ticker, which turned one `retrieval_node` call
  into an unbounded number of sequential round-trips — effectively a hang from the UI's
  perspective. Fixed two ways: `MAX_ARTICLES_PER_FETCH = 15` bounds how many articles get
  indexed per fetch, and `embed_batch()`/`upsert_batch()` were added to the `Embedder`/
  `VectorStore` protocols so N articles cost one embedding call + one Pinecone call instead of
  N of each. Measured effect on the 11-leg sample dataset: `retrieval_node` dropped from
  ~170–200s to ~45–56s.
- **Pinecone serverless index cap.** The free tier caps a project at 5 serverless indexes
  total, shared across every project using that Pinecone account — not per-project. Creating
  `trade-coach-news` and `trade-coach-concepts` required freeing a slot by deleting an unused
  index from an unrelated course project first; `trade-coach-patterns` still has no live index
  and falls back to `InMemoryVectorStore` until a slot frees up. `memory/create_index.py`
  creates whichever of `PINECONE_INDEX_NEWS`/`_CONCEPTS`/`_PATTERNS` are set and don't already
  exist, idempotently.
- **XSS gap in the Streamlit rewrite.** The Scorecard tab's redesign (see below) renders trade
  cards via `st.markdown(..., unsafe_allow_html=True)` for colored badges. `trade.ticker`
  comes straight from the uploaded CSV's `Instrument` column — unvalidated user input, not a
  closed vocabulary like verdict/status — so it was being interpolated into raw HTML
  unescaped. Fixed with `ui.escaped_ticker()` (`html.escape`); `ui.badge()` additionally
  rejects any `kind` outside its closed vocabulary, so only enum-controlled values can ever
  reach the unsafe-HTML path.

## Streamlit UI pass

`.streamlit/config.toml` sets an explicit fintech palette (deep emerald primary, warm
off-white background) instead of Streamlit's default theme. `app/ui.py` holds all
presentation logic — CSS plus small enum-keyed helpers (`badge()`, `confidence_badge()`,
`ticker_tape_html()`) — kept separate from `streamlit_app.py` specifically so the "what's safe
to interpolate into unsafe HTML" rule lives in one auditable place (see its module docstring).

- Scorecard tab: rewritten from a flat `st.dataframe` into per-trade cards — ticker + status
  badge, one row per leg with buy/sell/verdict badges, price, percentile — with news and coach
  narrative kept on the escaped `st.write`/`st.caption` path since both are external/LLM text.
- Patterns tab: each card leads with a HIGH/MODERATE/LOW CONFIDENCE badge.
- `st.logo("assets/logo.svg")` for a project mark in the header.
- A live-quoted, auto-scrolling ticker tape (`mcp_server/ticker_feed.py`, real-time via
  yfinance's `fast_info`) shows the uploaded trades' tickers, falling back to a default
  watchlist before any CSV is run. Wrapped in `@st.fragment(run_every=30)` — Streamlit's
  native periodic-rerun primitive (1.37+) — so only the tape re-renders every 30s instead of
  the whole page; no new dependency needed.

## Known limitations / roadmap (unchanged from the Product Spec, confirmed during the build)

- `mcp_server/client.py` spawns one MCP stdio subprocess per `get_local_extremes` call rather
  than reusing one session per run — correct but not efficient; fine for course-scope trade
  volumes, worth pooling before any real usage at scale. Directly observable in the live logs
  as one `CallToolRequest`/`ListToolsRequest` pair per leg.
- FIFO-only lot matching, single hardcoded user/namespace, no multi-broker support — all as
  scoped out in Product Spec §2.
- Finnhub's free-tier news lookback (~12 months, empirically measured, not documented) and
  Pinecone's 5-serverless-index account cap are both real ceilings on this integration, not
  just this app's code — see the live-integration section above.

## Running it

Run from this directory (`week6/`):

```bash
uv run pytest                        # full offline suite (90 tests)
uv run python -m evals.deterministic_eval
uv run python -m evals.groundedness_eval
uv run streamlit run app/streamlit_app.py
```

Fill in `.env` (copy from `.env.example`) with real `ANTHROPIC_API_KEY` / `PINECONE_API_KEY` /
`OPENAI_API_KEY` / `NEBIUS_API_KEY` / `NEWS_API_KEY` (a Finnhub key — see the news-provider
swap above) to switch each dependency from its offline fake to the real integration — no code
changes required. `PINECONE_API_KEY` alone isn't enough: the indexes themselves don't exist
until created, so before the first live run:

```bash
uv run python -m memory.create_index   # idempotent; creates whichever PINECONE_INDEX_* are set
```
