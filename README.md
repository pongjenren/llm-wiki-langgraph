# llm-wiki

An ingest pipeline that turns raw documents into a cross-referenced wiki.
[LangGraph](https://github.com/langchain-ai/langgraph) drives the flow; every LLM
step runs through [nanobot](https://github.com/HKUDS/nanobot), which calls
`openai/gpt-oss-120b` via OpenRouter.

## How it works

Ingest runs in two stages. The sketches below are the shape of the flow;
[`docs/graphs.md`](docs/graphs.md) has the exact graphs, generated from the
compiled code via `uv run llm-wiki graph`.

**Stage 1 — document → items** (`graph/doc_graph.py`)

```
raw file → load → SHA256 check ─┬─ already ingested → stop
                                └─ new → summarize? ─┬─ yes → summarize ─┐
                                                     └─ no ─────────────┴→ extract items
```

The SHA is taken over the file's raw bytes, so re-running never reprocesses an
unchanged document. Whether to summarize is the model's call: a spreadsheet or
log gets rewritten into descriptive prose first, while text that is already prose
goes straight to extraction. The source row is written only after extraction
succeeds, so a document that fails partway can be retried.

**Stage 2 — item → page** (`graph/item_graph.py`)

```
item → review ─fail→ refine ─┐
         │  ↑────────────────┘
         pass
         ↓
     does the entity exist?
      ├─ no  → insert page → reference [1] → create page
      └─ yes → link source → reference [N] → merge into existing page
                                     ↓
                          page review ─fail→ refine ─┐
                                │  ↑─────────────────┘
                                pass
                                ↓
                    write .md + index.md, update db
```

Items are processed one at a time. That is what makes reference numbering and
entity resolution safe — two items naming the same entity would otherwise race to
create two pages for it.

**Citations.** While an item is being processed we do not yet know which
reference number its source will get on the target page, so the model always
writes `[CURRENT]`. Once `wiki_source` assigns the number, `[CURRENT]` is
substituted for `[N]`. The References section itself is generated from the
database on every write, so the model never renumbers existing citations.

**Entity resolution** is two-stage: an exact lookup in `page_aliases` (names are
casefolded and whitespace-collapsed), then a vector search over page names. A
similarity hit is recorded as an `embedding_sim` alias, so the next lookup for
that name is an exact hit. Only the *name* is embedded — mixing the description
in measures topical similarity rather than identity, which was enough to split
"Self-attention network" and "Self-attention networks" into two pages.

**Merges are verified.** The merge step asks for a full rewrite of the page, so
the result is checked programmatically: every citation marker and every section
heading the old page carried must still be present. If not, the merge is
re-requested with the specific loss named.

**Review failures** never lose data: after the refine budget is spent the page is
still written, but flagged `needs_review` in the database, banner-marked in the
page, and starred in `index.md`.

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
`index.md` per namespace. The markdown files are the source of truth; SQLite
holds metadata only.

## Schema

SQLite with [sqlite-vec](https://github.com/asg017/sqlite-vec) for similarity
search (`db/schema.sql`).

| Table | Holds |
|---|---|
| `source` | ingested documents, keyed `UNIQUE(namespace, sha256)` |
| `wiki_pages` | page metadata, `UNIQUE(namespace, page_name)`, plus `needs_review` |
| `page_aliases` | names → page, PK `(namespace, query_name)`, typed `canonical` / `embedding_sim` / `manual` |
| `wiki_source` | which sources cite a page and in what order, `UNIQUE(wiki_id, source_id)` |
| `wiki_page_embeddings` | `vec0` virtual table; created at runtime because its dimension follows the embedding model |

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
  llm/            nanobot client, prompts, reply schemas
  db/             schema.sql, connection/bootstrap, query helpers
  loaders/        extension-dispatched text extraction
  pages.py        markdown page + index writing, merge validation
  embedding.py    local sentence-transformers
nanobot.config.json   provider, model preset, tools (all disabled)
```

Adding a file format means writing one function and registering it in
`loaders.LOADERS`; nothing else changes.
