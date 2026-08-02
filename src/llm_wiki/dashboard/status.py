"""Combine the raw/ filesystem with the `source` table into a per-file view."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm_wiki import loaders
from llm_wiki.db import repo
from llm_wiki.db.connection import Connection

# Dashboard-facing status of a raw file, in the order we want columns/badges to
# read. "pending" means present on disk but never ingested.
STATUS_PENDING = "pending"
STATUS_OK = "ok"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class RawFileStatus:
    namespace: str
    filename: str
    path: str
    status: str
    source_id: int | None = None
    seconds: float | None = None
    last_processed: datetime | None = None
    error_msg: str | None = None


@dataclass(frozen=True)
class StatusSummary:
    files: list[RawFileStatus]
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return len(self.files)


def compute_raw_status(conn: Connection, raw_dir: Path) -> StatusSummary:
    """Every supported file under raw/, annotated with its ingest outcome."""
    sources = repo.sources_by_namespace_filename(conn)

    files: list[RawFileStatus] = []
    counts = {STATUS_PENDING: 0, STATUS_OK: 0, STATUS_FAILED: 0}

    targets = loaders.iter_raw_files(raw_dir) if raw_dir.exists() else []
    for namespace, path in targets:
        source = sources.get((namespace, path.name))
        if source is None:
            status = STATUS_PENDING
            files.append(
                RawFileStatus(
                    namespace=namespace, filename=path.name, path=str(path), status=status
                )
            )
        else:
            status = source["status"]
            files.append(
                RawFileStatus(
                    namespace=namespace,
                    filename=path.name,
                    path=str(path),
                    status=status,
                    source_id=source["id"],
                    seconds=source["seconds"],
                    last_processed=source["ingest_time"],
                    error_msg=source["error_msg"],
                )
            )
        counts[status] = counts.get(status, 0) + 1

    files.sort(key=lambda f: (f.namespace, f.filename))
    return StatusSummary(files=files, counts=counts)
