"""Command-line entry point."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer

from llm_wiki import embedding, loaders
from llm_wiki.config import settings
from llm_wiki.db.connection import connect, init_db
from llm_wiki.graph import Deps
from llm_wiki.llm.client import NanobotClient
from llm_wiki.pipeline import DocumentOutcome, ingest_documents

app = typer.Typer(help="Ingest raw documents into the llm-wiki knowledge base.")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # nanobot logs through loguru, not stdlib logging, and is extremely chatty
    # at DEBUG — it drowns out pipeline progress entirely. Its default sink has
    # to be replaced rather than filtered.
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")


def _resolve_targets(path: Optional[Path], namespace: Optional[str]) -> list[tuple[str, Path]]:
    """Work out which (namespace, file) pairs to ingest."""
    if path is None:
        if not settings.raw_dir.exists():
            raise typer.BadParameter(f"raw directory does not exist: {settings.raw_dir}")
        targets = loaders.iter_raw_files(settings.raw_dir)
        if namespace is not None:
            targets = [(ns, p) for ns, p in targets if ns == namespace]
        return targets

    if path.is_dir():
        # A directory holding documents is a namespace; a directory holding only
        # other directories is a raw root. Both forms are worth accepting, since
        # "ingest raw/" and "ingest raw/ml" are equally natural to type.
        direct_files = [p for p in sorted(path.glob("*")) if loaders.is_supported(p)]
        if direct_files:
            return [(namespace or path.name, p) for p in loaders.iter_namespace_files(path)]
        targets = loaders.iter_raw_files(path)
        if namespace is not None:
            targets = [(ns, p) for ns, p in targets if ns == namespace]
        return targets

    if namespace is None:
        # raw/<namespace>/<file>: the parent directory names the namespace.
        namespace = path.parent.name
    return [(namespace, path)]


def _report(outcomes: list[DocumentOutcome]) -> int:
    """Print a summary and return the process exit code."""
    created = merged = skipped = flagged = failed = 0

    for outcome in outcomes:
        if outcome.error:
            failed += 1
            typer.secho(f"✗ {outcome.path}: {outcome.error}", fg=typer.colors.RED)
            continue
        if outcome.skipped:
            skipped += 1
            typer.secho(f"– {outcome.path}: {outcome.skip_reason}", fg=typer.colors.BRIGHT_BLACK)
            continue

        note = " (summarized)" if outcome.summarized else ""
        typer.secho(f"✓ {outcome.path}{note}", fg=typer.colors.GREEN)
        for item in outcome.items:
            if item.error:
                failed += 1
                typer.secho(f"    ✗ {item.name}: {item.error}", fg=typer.colors.RED)
                continue
            if item.is_new_page:
                created += 1
                action = "created"
            else:
                merged += 1
                action = f"merged as [{item.reference_number}]"
            flag = ""
            if item.needs_review:
                flagged += 1
                flag = " ⚠️ needs review"
            typer.echo(f"    · {item.page_name}: {action}{flag}")

    typer.echo("")
    typer.echo(
        f"{len(outcomes)} document(s): {created} page(s) created, {merged} merged, "
        f"{skipped} skipped, {flagged} flagged, {failed} failed."
    )
    return 1 if failed else 0


@app.command()
def ingest(
    path: Optional[Path] = typer.Argument(
        None,
        exists=True,
        help="File or directory to ingest. Defaults to the whole raw/ directory.",
    ),
    namespace: Optional[str] = typer.Option(
        None, "--namespace", "-n", help="Override the namespace inferred from the directory name."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logging."),
) -> None:
    """Ingest documents from raw/<namespace>/ into the wiki."""
    _configure_logging(verbose)

    if not settings.openrouter_api_key:
        typer.secho(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    targets = _resolve_targets(path, namespace)
    if not targets:
        typer.secho("Nothing to ingest.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.echo(f"Ingesting {len(targets)} document(s) into {settings.wiki_dir}\n")

    async def run() -> int:
        conn = connect(settings.db_path)
        try:
            init_db(conn, embedding.embedding_dim(model_name=settings.embedding_model))
            async with NanobotClient() as client:
                deps = Deps(conn=conn, client=client, settings=settings)
                outcomes = await ingest_documents(deps, targets)
            return _report(outcomes)
        finally:
            conn.close()

    raise typer.Exit(code=asyncio.run(run()))


@app.command()
def graph() -> None:
    """Print both graphs as mermaid, for docs/graphs.md."""
    from llm_wiki.graph import build_doc_graph, build_item_graph

    # Building a graph never runs a node, so the dependencies can be empty here.
    deps = Deps(conn=None, client=None, settings=settings)  # type: ignore[arg-type]
    for title, builder in (
        ("Stage 1 — `doc_graph`", build_doc_graph),
        ("Stage 2 — `item_graph`", build_item_graph),
    ):
        typer.echo(f"## {title}\n")
        typer.echo("```mermaid")
        typer.echo(builder(deps).get_graph().draw_mermaid().strip())
        typer.echo("```\n")


@app.command()
def status() -> None:
    """Show what is currently in the knowledge base."""
    conn = connect(settings.db_path)
    try:
        init_db(conn, embedding.embedding_dim(model_name=settings.embedding_model))
        namespaces = conn.execute(
            "SELECT namespace, COUNT(*) AS pages, SUM(needs_review) AS flagged "
            "FROM wiki_pages GROUP BY namespace ORDER BY namespace"
        ).fetchall()
        sources = conn.execute("SELECT COUNT(*) AS n FROM source").fetchone()["n"]

        if not namespaces:
            typer.echo("No pages yet.")
            return

        typer.echo(f"{sources} source document(s) ingested.\n")
        for row in namespaces:
            flagged = f", {row['flagged']} flagged" if row["flagged"] else ""
            typer.echo(f"  {row['namespace']}: {row['pages']} page(s){flagged}")
    finally:
        conn.close()


def main() -> None:
    app()
