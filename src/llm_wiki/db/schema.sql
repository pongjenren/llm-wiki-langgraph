-- Core ingest schema (PostgreSQL).

CREATE EXTENSION IF NOT EXISTS vector;

-- Raw ingested source documents, one row per document ingested. Also carries
-- that document's ingest outcome: there is no separate run/document telemetry
-- table, because a run is a single document.
CREATE TABLE IF NOT EXISTS source (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename    TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ,                    -- the document's own mtime
    namespace   TEXT        NOT NULL,
    -- Nullable: a document that failed before its content could be hashed
    -- (unreadable file, unsupported format) still gets a row.
    sha256      TEXT,
    ingest_time TIMESTAMPTZ NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'failed')),
    -- The document-level error, or the names of the items that failed. NULL
    -- when status = 'ok'; successful items are not recorded.
    error_msg   TEXT,
    seconds     DOUBLE PRECISION,               -- wall time for this document
    -- Dedup is per namespace: the same file ingested into two namespaces is
    -- two independent sources. A failed row occupies the key too -- a document
    -- is attempted once, and retrying it means deleting its row first.
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
    -- Name embedding for entity resolution, dimension fixed to the deployed
    -- model (NV-Embed-v2 = 4096). Nullable: a page can exist before it is
    -- embedded. No ANN index -- pgvector's hnsw/ivfflat cap at 2000 dims -- so
    -- resolution does an exact scan, fine at the page counts here. Change the
    -- model's width? Change it here.
    embedding    vector(4096),
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
