"""Combine the raw/ filesystem with DB telemetry into a per-file status view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_wiki import loaders
from llm_wiki.db import repo
from llm_wiki.db.connection import Connection

# Dashboard-facing status of a raw file, in the order we want columns/badges to
# read. "pending" means present on disk but never seen by an ingest run.
STATUS_PENDING = "pending"
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class RawFileStatus:
    namespace: str
    filename: str
    path: str
    status: str
    seconds: float | None = None
    last_processed: str | None = None
    skip_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class StatusSummary:
    files: list[RawFileStatus]
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return len(self.files)


def compute_raw_status(conn: Connection, raw_dir: Path) -> StatusSummary:
    """Every supported file under raw/, annotated with its last-known outcome."""
    docs = repo.latest_ingest_docs(conn)
    # Paths are canonicalized on write, so the resolved path is the precise key.
    # The filename index is a fallback for rows written before that (they stored
    # whatever path the CLI was given, e.g. a cwd-relative one). It is looser —
    # two same-named files in different sub-directories of one namespace share a
    # key — so it is only consulted when the exact path lookup misses.
    by_path = {(d["namespace"], d["path"]): d for d in docs}
    by_filename = {(d["namespace"], d["filename"]): d for d in docs}
    known_sources = repo.source_filenames_by_namespace(conn)

    files: list[RawFileStatus] = []
    counts = {STATUS_PENDING: 0, STATUS_OK: 0, STATUS_SKIPPED: 0, STATUS_FAILED: 0}

    targets = loaders.iter_raw_files(raw_dir) if raw_dir.exists() else []
    for namespace, path in targets:
        doc = by_path.get((namespace, str(path.resolve()))) or by_filename.get(
            (namespace, path.name)
        )
        if doc is not None:
            status = doc["status"]
            files.append(
                RawFileStatus(
                    namespace=namespace,
                    filename=path.name,
                    path=str(path),
                    status=status,
                    seconds=doc["seconds"],
                    last_processed=doc["run_finished_at"] or doc["run_started_at"],
                    skip_reason=doc["skip_reason"],
                    error=doc["error"],
                )
            )
        elif (namespace, path.name) in known_sources:
            # Ingested before telemetry existed: no run/doc row, but a source
            # exists. Treat as done, without timing.
            status = STATUS_OK
            files.append(
                RawFileStatus(
                    namespace=namespace, filename=path.name, path=str(path), status=status
                )
            )
        else:
            status = STATUS_PENDING
            files.append(
                RawFileStatus(
                    namespace=namespace, filename=path.name, path=str(path), status=status
                )
            )
        counts[status] = counts.get(status, 0) + 1

    files.sort(key=lambda f: (f.namespace, f.filename))
    return StatusSummary(files=files, counts=counts)
