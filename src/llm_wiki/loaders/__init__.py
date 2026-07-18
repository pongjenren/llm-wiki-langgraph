"""Raw file loaders.

Loaders turn a file into deterministic text. This is deliberately kept out of
the LLM's hands: the SHA256 dedup check is only meaningful if the same bytes
always produce the same text, so extraction must be reproducible. Interpreting
that text (e.g. narrating a spreadsheet) is the summarize node's job.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from llm_wiki.loaders.text import load_text
from llm_wiki.loaders.xlsx import load_xlsx


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    text: str
    sha256: str
    timestamp: str


# Extension -> loader. Add new formats here; nothing else needs to change.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".md": load_text,
    ".txt": load_text,
    ".xlsx": load_xlsx,
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


class UnsupportedFileType(Exception):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"No loader for {path.suffix or '(no extension)'}: {path}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        self.path = path


def load(path: Path) -> LoadedDocument:
    """Read a file into text plus the hash of its raw bytes.

    The hash covers the original bytes, not the extracted text, so changing the
    extraction logic never makes an already-ingested document look new.
    """
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileType(path)

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return LoadedDocument(path=path, text=loader(path), sha256=sha256, timestamp=timestamp)


def is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def iter_namespace_files(namespace_dir: Path) -> list[Path]:
    """Every supported file inside one namespace directory."""
    return [p for p in sorted(namespace_dir.rglob("*")) if is_supported(p)]


def iter_raw_files(raw_dir: Path) -> list[tuple[str, Path]]:
    """Yield (namespace, path) for every supported file under raw/<namespace>/."""
    found: list[tuple[str, Path]] = []
    for namespace_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        found += [(namespace_dir.name, path) for path in iter_namespace_files(namespace_dir)]
    return found
