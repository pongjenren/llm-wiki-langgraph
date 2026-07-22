-- Core ingest schema (PostgreSQL). The vector table is created separately (see
-- connection.py) because its dimension depends on the configured embedding model.

CREATE EXTENSION IF NOT EXISTS vector;

-- Raw ingested source documents.
CREATE TABLE IF NOT EXISTS source (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename    TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ,                    -- the document's own mtime
    namespace   TEXT        NOT NULL,
    sha256      TEXT        NOT NULL,
    ingest_time TIMESTAMPTZ NOT NULL,
    -- Dedup is per namespace: the same file ingested into two namespaces is
    -- two independent sources.
    UNIQUE (namespace, sha256)
);

-- Core entity/concept pages. Page bodies live on disk as markdown; this table
-- only carries metadata.
CREATE TABLE IF NOT EXISTS wiki_pages (
    page_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    page_name    TEXT        NOT NULL,          -- display name
    namespace    TEXT        NOT NULL,
    type         TEXT        NOT NULL CHECK (type IN ('entity', 'concept')),
    needs_review BOOLEAN     NOT NULL DEFAULT FALSE,
    create_time  TIMESTAMPTZ NOT NULL,
    UNIQUE (namespace, page_name)
);

-- Alternate names that resolve to a canonical page, used by the
-- "does this entity already exist?" lookup. query_name is normalized
-- (casefolded, whitespace-collapsed); page_name keeps the display form.
CREATE TABLE IF NOT EXISTS page_aliases (
    namespace  TEXT   NOT NULL,
    query_name TEXT   NOT NULL,
    page_id    BIGINT NOT NULL REFERENCES wiki_pages (page_id) ON DELETE CASCADE,
    type       TEXT   NOT NULL CHECK (type IN ('canonical', 'alias', 'embedding_sim', 'string_sim', 'llm_sim', 'manual')),
    PRIMARY KEY (namespace, query_name)
);

CREATE INDEX IF NOT EXISTS idx_page_aliases_page ON page_aliases (page_id);

-- One row per `llm-wiki ingest` invocation. Powers the dashboard's run
-- history and the "total ingest time" figure. Independent of `source`: a run
-- is recorded even when every document is skipped or fails.
CREATE TABLE IF NOT EXISTS ingest_run (
    run_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ,                    -- NULL while a run is in flight
    total_seconds DOUBLE PRECISION,
    doc_count     INTEGER NOT NULL DEFAULT 0,
    created       INTEGER NOT NULL DEFAULT 0,     -- pages created across the run
    merged        INTEGER NOT NULL DEFAULT 0,     -- pages merged into
    skipped       INTEGER NOT NULL DEFAULT 0,     -- documents skipped (e.g. dup)
    flagged       INTEGER NOT NULL DEFAULT 0,     -- items flagged needs_review
    failed        INTEGER NOT NULL DEFAULT 0      -- documents or items that errored
);

-- One row per document processed within a run. `status` is the document-level
-- outcome; item-level detail (created/merged/flagged/errors) is kept as JSONB in
-- `items_json` so the dashboard can render a per-file log without another table.
CREATE TABLE IF NOT EXISTS ingest_doc (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id       BIGINT  NOT NULL REFERENCES ingest_run (run_id) ON DELETE CASCADE,
    namespace    TEXT    NOT NULL,
    filename     TEXT    NOT NULL,
    path         TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('ok', 'skipped', 'failed')),
    seconds      DOUBLE PRECISION,
    skip_reason  TEXT,
    error        TEXT,
    items_json   JSONB   NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ingest_doc_run ON ingest_doc (run_id);
CREATE INDEX IF NOT EXISTS idx_ingest_doc_path ON ingest_doc (namespace, path);

-- Which sources contributed to a page, and in what citation order.
CREATE TABLE IF NOT EXISTS wiki_source (
    link_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    wiki_id         BIGINT  NOT NULL REFERENCES wiki_pages (page_id) ON DELETE CASCADE,
    source_id       BIGINT  NOT NULL REFERENCES source (id) ON DELETE CASCADE,
    reference_order INTEGER NOT NULL,
    namespace       TEXT    NOT NULL,
    -- A source cites a given page exactly once, and reference numbers are
    -- unique within a page.
    UNIQUE (wiki_id, source_id),
    UNIQUE (wiki_id, reference_order)
);

CREATE INDEX IF NOT EXISTS idx_wiki_source_wiki ON wiki_source (wiki_id);

-- Directed page-to-page links discovered in page bodies: a mention of one
-- page's name inside another page's body becomes a markdown link, recorded
-- here. One row per (source, target) pair -- a page links to a given target at
-- most once (its first mention) -- and re-linking a page replaces all of its
-- rows. Both endpoints are in the same namespace.
CREATE TABLE IF NOT EXISTS wiki_links (
    src_page_id      BIGINT NOT NULL REFERENCES wiki_pages (page_id) ON DELETE CASCADE,
    dst_page_id      BIGINT NOT NULL REFERENCES wiki_pages (page_id) ON DELETE CASCADE,
    namespace        TEXT   NOT NULL,
    anchor_text      TEXT   NOT NULL,   -- the surface text that got linked
    context_sentence TEXT,             -- the whole sentence the mention sits in
    PRIMARY KEY (src_page_id, dst_page_id)
);

-- Backlinks: "which pages link to this one?"
CREATE INDEX IF NOT EXISTS idx_wiki_links_dst ON wiki_links (dst_page_id);
