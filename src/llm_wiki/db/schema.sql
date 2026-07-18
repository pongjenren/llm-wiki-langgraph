-- Core ingest schema. The vector table is created separately (see connection.py)
-- because its dimension depends on the configured embedding model.

PRAGMA foreign_keys = ON;

-- Raw ingested source documents.
CREATE TABLE IF NOT EXISTS source (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    timestamp   TEXT,                     -- the document's own mtime
    namespace   TEXT    NOT NULL,
    sha256      TEXT    NOT NULL,
    ingest_time TEXT    NOT NULL,
    -- Dedup is per namespace: the same file ingested into two namespaces is
    -- two independent sources.
    UNIQUE (namespace, sha256)
);

-- Core entity/concept pages. Page bodies live on disk as markdown; this table
-- only carries metadata.
CREATE TABLE IF NOT EXISTS wiki_pages (
    page_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    page_name    TEXT    NOT NULL,        -- display name
    namespace    TEXT    NOT NULL,
    type         TEXT    NOT NULL CHECK (type IN ('entity', 'concept')),
    needs_review INTEGER NOT NULL DEFAULT 0,
    create_time  TEXT    NOT NULL,
    UNIQUE (namespace, page_name)
);

-- Alternate names that resolve to a canonical page, used by the
-- "does this entity already exist?" lookup. query_name is normalized
-- (casefolded, whitespace-collapsed); page_name keeps the display form.
CREATE TABLE IF NOT EXISTS page_aliases (
    namespace  TEXT    NOT NULL,
    query_name TEXT    NOT NULL,
    page_id    INTEGER NOT NULL REFERENCES wiki_pages (page_id) ON DELETE CASCADE,
    type       TEXT    NOT NULL CHECK (type IN ('canonical', 'embedding_sim', 'manual')),
    PRIMARY KEY (namespace, query_name)
);

CREATE INDEX IF NOT EXISTS idx_page_aliases_page ON page_aliases (page_id);

-- Which sources contributed to a page, and in what citation order.
CREATE TABLE IF NOT EXISTS wiki_source (
    link_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    wiki_id         INTEGER NOT NULL REFERENCES wiki_pages (page_id) ON DELETE CASCADE,
    source_id       INTEGER NOT NULL REFERENCES source (id) ON DELETE CASCADE,
    reference_order INTEGER NOT NULL,
    namespace       TEXT    NOT NULL,
    -- A source cites a given page exactly once, and reference numbers are
    -- unique within a page.
    UNIQUE (wiki_id, source_id),
    UNIQUE (wiki_id, reference_order)
);

CREATE INDEX IF NOT EXISTS idx_wiki_source_wiki ON wiki_source (wiki_id);
