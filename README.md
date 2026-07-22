# llm-wiki

An ingest pipeline that turns raw documents into a cross-referenced wiki.
[LangGraph](https://github.com/langchain-ai/langgraph) drives the flow; every LLM
step calls `openai/gpt-oss-120b` directly via OpenRouter (`llm_wiki.llm.query_LLM`).

## How it works

Ingest runs in two stages. The sketches below are the shape of the flow;
[`docs/graphs.md`](docs/graphs.md) has the exact graphs, generated from the
compiled code via `uv run llm-wiki graph`.

**Stage 1 — document → items** (`graph/doc_graph.py`)

```
raw file → load → SHA256 check ─┬─ already ingested → stop
                                └─ new → summarize? ─┬─ yes → summarize ─┐
                                                     └─ no ──────────────┴→ extract
                                                                             ↓
                                            review extraction ─fail→ refine ─┐
                                                   │  ↑──────────────────────┘
                                                   pass → record source → items
```

The SHA is taken over the file's raw bytes, so re-running never reprocesses an
unchanged document. Whether to summarize is the model's call: a spreadsheet or
log gets rewritten into descriptive prose first, while text that is already prose
goes straight to extraction. Extraction quality is settled here, against the
source: `review_extraction` checks the whole item list for recall, precision,
type, and aliases, and `refine_extraction` rebuilds it — adding missing items,
**dropping** spurious ones, and fixing the rest. The source row is written only
after this review succeeds, so a document that fails partway can be retried.

**Stage 2 — item → page** (`graph/item_graph.py`)

```
item → does the entity exist?
      ├─ no  → insert page → reference [1] → create page
      └─ yes → link source → reference [N] → merge into existing page
                                     ↓
                          page review ─fail→ refine ─┐
                                │  ↑─────────────────┘
                                pass
                                ↓
                    write .md + index.md, update db
```

Items arrive already vetted by stage 1, so stage 2 has no per-item review and
goes straight to entity resolution. Items are processed one at a time. That is
what makes reference numbering and entity resolution safe — two items naming the
same entity would otherwise race to create two pages for it.

**Citations.** Every fact in an item's description comes from the one document it
was extracted from, so the reference number is uniform across the description and
is not known until the source is linked to a page. Extraction therefore produces
plain facts with no markers; `create_page` / `merge_page` are told the assigned
number and attach `[N]` to the claims they write. The References section itself
is generated from the database on every write, so the model never renumbers
existing citations.

**Entity resolution** is a funnel. First an exact lookup in `page_aliases` (names
are casefolded and whitespace-collapsed). On a miss, candidates are gathered from
*two* signals — a vector search over page names (only the *name* is embedded;
mixing the description in measures topical similarity rather than identity) and
string matching over existing aliases (rapidfuzz `token_sort_ratio`, which catches
typos, plurals and casing the embeddings rank too far apart). A near-exact match
on either signal is accepted outright; otherwise the candidates, together with the
item's aliases and description, are handed to an LLM that decides whether it is the
same entity or a new one. However a name resolves, it is recorded as an alias
(`embedding_sim` / `string_sim` / `llm_sim`) so the next lookup for that name is an
exact hit — which also lets the LLM's verdicts converge over time. A low-confidence
LLM match still merges but flags the page for review.

**Merges are verified.** The merge step asks for a full rewrite of the page, so
the result is checked programmatically: every citation marker and every section
heading the old page carried must still be present. If not, the merge is
re-requested with the specific loss named.

**Review failures** never lose data: after the refine budget is spent the page is
still written, but flagged `needs_review` in the database, banner-marked in the
page, and starred in `index.md`.

**Cross-linking.** Once pages exist, a mention of one page inside another (same
namespace) becomes a markdown link — `TSMC` → `[TSMC](TSMC.md)` — recorded in
`wiki_links`. Matching scans the `page_aliases` dictionary with an Aho-Corasick
automaton, so per-page cost follows the page length, not the size of the
knowledge base. It is case- and whitespace-insensitive, respects word
boundaries, prefers the longest alias when several overlap, links each target at
its first mention only, and skips code, headings, and existing links. Ingest
links incrementally — only the pages a run touched, so they cross-link to each
other and to any existing page they mention. `llm-wiki link` relinks a whole
namespace to backfill links *into* pages that were added after their mentions
were written. Re-linking is idempotent: sibling-page links are unwrapped before
each pass, so pages are always relinked from a clean body.

## Setup

```bash
uv sync
cp .env.example .env      # then fill in OPENROUTER_API_KEY
```

Everything else has a default; see `.env.example` for the knobs.

## Usage

Drop files into `raw/<namespace>/`. The directory name is the namespace.
Supported: `.md`, `.txt`, `.xlsx`.

```bash
uv run llm-wiki ingest                      # everything under raw/
uv run llm-wiki ingest raw/ml               # one namespace
uv run llm-wiki ingest raw/ml/paper.md      # one file
uv run llm-wiki ingest notes.md -n research # override the namespace
uv run llm-wiki status                      # what is in the knowledge base
```

Pages are written to `wiki/<namespace>/<Page_Name>.md`, with a generated
`index.md` per namespace. The markdown files are the source of truth; PostgreSQL
holds metadata only.

Ingest cross-links the pages it touches automatically. Run a full reconcile when
you want links backfilled into pages added after their mentions were written:

```bash
uv run llm-wiki link                        # relink every namespace
uv run llm-wiki link demo                   # one namespace
uv run llm-wiki link demo --dry-run         # report changes without writing
```

A read-only web dashboard shows run history and per-file ingest status:

```bash
uv run llm-wiki dashboard                    # http://127.0.0.1:8000
uv run llm-wiki dashboard --port 9000        # bind a different port
```

To start over, `reset` empties `wiki/` and drops every llm-wiki table from the
database. `raw/` is never touched, so a following `ingest` rebuilds everything
from the same sources:

```bash
uv run llm-wiki reset                        # prompts before deleting
uv run llm-wiki reset --yes                  # skip the confirmation
```

## Schema

PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) for similarity
search (`db/schema.sql`). The database server is expected to run separately;
point `LLM_WIKI_DB_URL` at it (default
`postgresql://llm_wiki:llm_wiki@localhost:5432/llm_wiki`).

| Table | Holds |
|---|---|
| `source` | ingested documents, keyed `UNIQUE(namespace, sha256)` |
| `wiki_pages` | page metadata, `UNIQUE(namespace, page_name)`, plus `needs_review` |
| `page_aliases` | names → page, PK `(namespace, query_name)`, typed `canonical` / `embedding_sim` / `manual` |
| `wiki_source` | which sources cite a page and in what order, `UNIQUE(wiki_id, source_id)` |
| `wiki_links` | directed page → page links, PK `(src_page_id, dst_page_id)`, indexed by target for backlinks |
| `wiki_page_embeddings` | pgvector `vector(N)` table with an HNSW cosine index; created at runtime because its dimension follows the embedding model |
| `ingest_run` / `ingest_doc` | per-run and per-document ingest telemetry, powering the dashboard |

## Development

```bash
uv run pytest
```

Tests script every LLM reply, so the suite needs no API key and no network.

## Layout

```
src/llm_wiki/
  cli.py          typer entry point
  pipeline.py     drives both graphs per document
  config.py       env-backed settings
  graph/          doc_graph (stage 1), item_graph (stage 2), shared state
  llm/            OpenRouter client, prompts, reply schemas
  db/             schema.sql, connection/bootstrap, query helpers
  loaders/        extension-dispatched text extraction
  pages.py        markdown page + index writing, merge validation
  links.py        cross-page link detection, wiki_links
  embedding.py    local sentence-transformers
  telemetry.py    records ingest runs/docs for the dashboard
  dashboard/      read-only ingest status web page (starlette + jinja2)
```

Adding a file format means writing one function and registering it in
`loaders.LOADERS`; nothing else changes.
