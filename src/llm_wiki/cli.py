"""Command-line entry point."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import typer

from llm_wiki import embedding, links, loaders, telemetry
from llm_wiki.config import PROJECT_ROOT, settings
from llm_wiki.db import repo
from llm_wiki.db.connection import connect, init_db
from llm_wiki.graph import Deps
from llm_wiki.llm.client import LLMClient
from llm_wiki.pipeline import DocumentOutcome, ingest_documents

app = typer.Typer(help="Ingest raw documents into the llm-wiki knowledge base.")

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "openai", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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


def _link_touched_pages(conn, outcomes: list[DocumentOutcome]) -> None:
    """Relink the pages this run created or merged into.

    Incremental by design: only pages touched here are rescanned, so they link
    to one another and to any existing page they mention. Links *into* a new page
    from pages left untouched are the job of `llm-wiki link` (full reconcile).
    Best-effort -- a linking failure must not fail an otherwise good ingest.
    """
    touched: dict[str, set[int]] = {}
    for outcome in outcomes:
        for item in outcome.items:
            if item.error is None and item.page_id is not None:
                touched.setdefault(outcome.namespace, set()).add(item.page_id)

    for namespace, page_ids in touched.items():
        try:
            links.link_pages(
                conn, wiki_dir=settings.wiki_dir, namespace=namespace, page_ids=sorted(page_ids)
            )
        except Exception as exc:
            log.warning("failed to link pages in namespace %s: %s", namespace, exc)


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
            start = time.monotonic()
            client = LLMClient()
            deps = Deps(conn=conn, client=client, settings=settings)
            outcomes = await ingest_documents(deps, targets)
            elapsed = time.monotonic() - start
            try:
                telemetry.record_run(conn, outcomes, elapsed)
            except Exception as exc:  # telemetry must never fail an ingest
                log.warning("failed to record ingest telemetry: %s", exc)
            _link_touched_pages(conn, outcomes)
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


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Wipe the wiki pages and the database, back to a clean slate.

    Raw source documents are never touched — re-running `ingest` rebuilds
    everything from them.
    """
    wiki_dir = settings.wiki_dir
    db_path = settings.db_path

    # Guard against a misconfigured LLM_WIKI_WIKI_DIR turning this into an
    # rm -rf of the project (or of the raw corpus we promise not to touch).
    for forbidden, label in ((settings.raw_dir, "raw_dir"), (PROJECT_ROOT, "project root")):
        if wiki_dir == forbidden:
            typer.secho(f"Refusing to reset: wiki_dir is the {label}.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    pages = sorted(p for p in wiki_dir.rglob("*") if p.is_file()) if wiki_dir.exists() else []
    # SQLite in WAL mode keeps state in sidecar files; leaving them behind
    # would resurrect part of the old database.
    db_files = [p for p in (db_path, *(db_path.with_name(db_path.name + s) for s in ("-wal", "-shm"))) if p.exists()]

    if not pages and not db_files:
        typer.echo("Already clean — nothing to remove.")
        return

    typer.echo(f"This will delete {len(pages)} page(s) under {wiki_dir}")
    typer.echo(f"and {len(db_files)} database file(s) at {db_path}.")
    typer.secho(f"{settings.raw_dir} will not be touched.", fg=typer.colors.GREEN)
    if not yes:
        typer.confirm("Proceed?", abort=True)

    if wiki_dir.exists():
        shutil.rmtree(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    for path in db_files:
        path.unlink()

    typer.secho("Reset complete. Run `llm-wiki ingest` to rebuild.", fg=typer.colors.GREEN)


@app.command()
def link(
    namespace: Optional[str] = typer.Argument(
        None, help="Namespace to relink. Defaults to every namespace."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing anything."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logging."),
) -> None:
    """Rebuild cross-page links across a whole namespace (full reconcile).

    Ingest links incrementally as it runs; use this to backfill links into pages
    added after their mentions were written, or after bulk changes.
    """
    _configure_logging(verbose)
    conn = connect(settings.db_path)
    try:
        init_db(conn, embedding.embedding_dim(model_name=settings.embedding_model))

        if namespace is not None:
            namespaces = [namespace]
        else:
            namespaces = [
                row["namespace"]
                for row in conn.execute(
                    "SELECT DISTINCT namespace FROM wiki_pages ORDER BY namespace"
                )
            ]
        if not namespaces:
            typer.echo("No pages to link.")
            return

        total_changed = 0
        for ns in namespaces:
            page_ids = [row["page_id"] for row in repo.list_pages(conn, ns)]
            results = links.link_pages(
                conn,
                wiki_dir=settings.wiki_dir,
                namespace=ns,
                page_ids=page_ids,
                dry_run=dry_run,
            )
            changed = [r for r in results if r.changed]
            total_changed += len(changed)
            link_count = sum(len(r.links) for r in results)
            verb = "would update" if dry_run else "updated"
            typer.echo(
                f"{ns}: {verb} {len(changed)} of {len(results)} page(s), "
                f"{link_count} link(s) total."
            )
            for r in changed:
                typer.secho(f"    · {r.page_name}: {len(r.links)} link(s)", fg=typer.colors.CYAN)

        if dry_run and total_changed:
            typer.echo("\nDry run: no files were written.")
    finally:
        conn.close()


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Host to bind the dashboard server to."),
    port: int = typer.Option(8000, help="Port to serve the dashboard on."),
) -> None:
    """Serve the read-only ingest dashboard as a local web page."""
    from llm_wiki.dashboard import serve

    typer.echo(f"Ingest dashboard on http://{host}:{port}  (Ctrl-C to stop)")
    serve(host=host, port=port)


def main() -> None:
    app()
