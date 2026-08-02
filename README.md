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
                                └─ new → extract
                                            ↓
                           review extraction ─fail→ refine ─┐
                                  │  ↑──────────────────────┘
                                  pass → record source → assign pages
```

The SHA is taken over the file's raw bytes, so re-running never reprocesses an
unchanged document. Extraction quality is settled here, against the
source: `review_extraction` checks the whole item list for recall, precision,
type, and aliases, and `refine_extraction` rebuilds it — adding missing items,
**dropping** spurious ones, and fixing the rest. The source row is written only
after this review succeeds, so a document that fails partway can be retried.

`assign_pages` closes the stage by resolving every item to the page it belongs
to, so stage 2 receives one assignment per *page* rather than per item. It is
the one step that must run serially — it is where "does this page already
exist?" is answered, and two items naming the same thing must not both answer
"no". Items are therefore resolved against the pages this document has already
claimed as well as against the database, and two items that land on the same
page are folded into one before any page is written. Nothing is written to the
database here: the assignment is a plan, so an item that never gets a body does
not leave an empty page behind.

**Stage 2 — item → page** (`graph/item_graph.py`)

```
assignment → is this page new?
      ├─ yes → insert page → reference [1] → create page
      └─ no  → link source → reference [N] → merge into existing page
                                     ↓
                          page review ─fail→ refine ─┐
                                │  ↑─────────────────┘
                                pass
                                ↓
                          write .md, update db
```

Items arrive already vetted *and* already resolved by stage 1, so stage 2 has no
per-item review and no resolution step — it is handed a page and fills it in.

**A document's items are folded in concurrently** (`LLM_WIKI_ITEM_WORKERS`,
default 4; set it to 1 for serial ingest). This is safe precisely because
resolution happened in stage 1: every assignment owns a distinct page, so no two
workers contend for a page row, its reference numbering, or its file on disk.
The one shared resource left is the database connection, which a re-entrant lock
serializes — cheap, because no LLM call ever happens inside a transaction. The
namespace index covers every page, so it is written once per document after all
of its items are in, rather than once per item.

**Citations.** Every fact in an item's description comes from the one document it
was extracted from, so the reference number is uniform across the description and
is not known until the source is linked to a page. Extraction therefore produces
plain facts with no markers; `create_page` / `merge_page` are told the assigned
number and attach `[N]` to the claims they write. The References section itself
is generated from the database on every write, so the model never renumbers
existing citations.

**Entity resolution** is a funnel. First an exact lookup in `page_aliases` (names
are casefolded and whitespace-collapsed). On a miss, candidates are gathered from
*three* signals — a vector search over page names (only the *name* is embedded;
mixing the description in measures topical similarity rather than identity),
string matching over existing aliases (rapidfuzz `token_sort_ratio`, which catches
typos, plurals and casing the embeddings rank too far apart), and the pages this
same document has already claimed but not yet written. That last one matters
because resolution now runs ahead of any page being created: without it, a
document naming one new thing two ways would be told "no such page" twice and
produce a duplicate. Pending pages are offered to the judge under negative ids,
so the sign of its answer says which list it points into. The candidates,
together with the item's aliases and description, are handed to an LLM that decides
whether it is the same entity or a new one. However a name resolves, it is recorded as an alias
(`llm_sim`) so the next lookup for that name is an
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
namespace) can become a markdown link — `TSMC` → `[TSMC](TSMC.md)` — recorded in
`wiki_links`. An alias match no longer links on its own; it only nominates a
candidate. Each page being linked is handed two candidate lists: the pages
written from the *same source document* (strong references), and the pages whose
aliases appear in its body (an Aho-Corasick scan of the `page_aliases`
dictionary, so this cost follows the page length, not the size of the knowledge
base). An LLM judges which mentions are genuine cross-references. Its picks are
then verified and applied *without* the model: every target must be a real
candidate page, and each approved anchor is spliced by the same matcher — case-
and whitespace-insensitive, respecting word boundaries and protected regions
(code, headings, existing links), longest anchor wins, each target linked at its
first mention only. Ingest links incrementally — only the pages a run touched, so
they cross-link to each other and to any existing page they mention. `llm-wiki
link` relinks a whole namespace to backfill links *into* pages that were added
after their mentions were written (no shared-source list there, only the alias
candidates). Re-linking is idempotent: sibling-page links are unwrapped before
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

When a document mostly lands but a handful of its items fail — a provider
hiccup, a timeout — `retry` re-runs just those. It reads the failed item names
out of `source.error_msg`, re-extracts the document, and folds only those items
into their pages under the document's original source row: pages that already
succeeded are not rewritten and citations keep their reference numbers. Clearing
every recorded failure sets the source back to `ok`.

```bash
uv run llm-wiki retry                       # every failed source
uv run llm-wiki retry 12 15                 # specific source ids
uv run llm-wiki retry -n ml                 # only one namespace
```

A document that failed outright — nothing extracted, no items — is not retried
this way; delete its `source` row and run `ingest` again. A raw file edited since
it was ingested is refused too, since re-extracting it would attribute new
material to the old document's hash.

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
| `source` | ingested documents, keyed `UNIQUE(namespace, sha256)`, carrying each document's ingest outcome (`status`, `error_msg`, `seconds`) |
| `wiki_pages` | page metadata, `UNIQUE(namespace, page_name)`, plus `needs_review` |
| `page_aliases` | names → page, PK `(namespace, query_name)`, typed `canonical` / `embedding_sim` / `manual` |
| `wiki_source` | which sources cite a page and in what order, `UNIQUE(wiki_id, source_id)` |
| `wiki_links` | directed page → page links, PK `(src_page_id, dst_page_id)`, indexed by target for backlinks |

Page name embeddings live on `wiki_pages.embedding` (`vector(4096)`, matching
NV-Embed-v2). There is no separate ingest telemetry table: one ingest is one
document, so the `source` row is what the dashboard reads.

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
