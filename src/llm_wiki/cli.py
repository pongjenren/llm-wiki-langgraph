"""Command-line entry point."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

import typer

from llm_wiki import links, loaders, telemetry
from llm_wiki.config import PROJECT_ROOT, settings
from llm_wiki.db import repo
from llm_wiki.db.connection import connect, init_db
from llm_wiki.graph import Deps
from llm_wiki.llm.client import LLMClient
from llm_wiki.pipeline import DocumentOutcome, RetryOutcome, ingest_documents, retry_sources

app = typer.Typer(help="Ingest raw documents into the llm-wiki knowledge base.")

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "openai"):
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


def _link_touched_pages(
    conn, client, outcomes: Sequence[DocumentOutcome | RetryOutcome]
) -> None:
    """Relink the pages this run created or merged into.

    Incremental by design: only pages touched here are rescanned, so they link
    to one another and to any existing page they mention. Links *into* a new page
    from pages left untouched are the job of `llm-wiki link` (full reconcile).
    Best-effort -- a linking failure must not fail an otherwise good ingest.

    Pages written from the same document are each other's first-list link
    candidates, so they are grouped per namespace as ``page_id -> siblings``.
    """
    touched: dict[str, dict[int, set[int]]] = {}
    for outcome in outcomes:
        page_ids = {
            item.page_id
            for item in outcome.items
            if item.error is None and item.page_id is not None
        }
        if not page_ids:
            continue
        siblings = touched.setdefault(outcome.namespace, {})
        for page_id in page_ids:
            siblings.setdefault(page_id, set()).update(page_ids - {page_id})

    for namespace, siblings in touched.items():
        try:
            links.link_pages(
                conn,
                wiki_dir=settings.wiki_dir,
                namespace=namespace,
                page_ids=sorted(siblings),
                client=client,
                source_siblings=siblings,
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

        typer.secho(f"✓ {outcome.path}", fg=typer.colors.GREEN)
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
            # More than one name means the document named this page twice and
            # the two were folded together before the page was written.
            folded = ""
            if len(item.merged_names) > 1:
                folded = f" (+{', '.join(item.merged_names[1:])})"
            typer.echo(f"    · {item.page_name}{folded}: {action}{flag}")

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

    def run() -> int:
        conn = connect(settings.db_url)
        try:
            init_db(conn)
            client = LLMClient()
            deps = Deps(conn=conn, client=client, settings=settings)
            outcomes = ingest_documents(deps, targets)
            try:
                telemetry.record_documents(conn, outcomes)
            except Exception as exc:  # telemetry must never fail an ingest
                log.warning("failed to record ingest telemetry: %s", exc)
            _link_touched_pages(conn, client, outcomes)
            return _report(outcomes)
        finally:
            conn.close()

    raise typer.Exit(code=run())


def _report_retries(outcomes: list[RetryOutcome]) -> int:
    """Print a retry summary and return the process exit code."""
    fixed = still_failing = flagged = 0

    for outcome in outcomes:
        label = f"source #{outcome.source_id} {outcome.filename}"

        if outcome.error:
            still_failing += len(outcome.requested) or 1
            typer.secho(f"✗ {label}: {outcome.error}", fg=typer.colors.RED)
            continue

        typer.secho(f"↻ {label}: retrying {len(outcome.requested)} item(s)", fg=typer.colors.CYAN)
        if outcome.collateral:
            typer.secho(
                f"    ⚠️ also re-merging {', '.join(outcome.collateral)}, which did not fail",
                fg=typer.colors.YELLOW,
            )

        for item in outcome.items:
            if item.error:
                still_failing += 1
                typer.secho(f"    ✗ {item.name}: {item.error}", fg=typer.colors.RED)
                continue
            fixed += 1
            action = "created" if item.is_new_page else f"merged as [{item.reference_number}]"
            flag = ""
            if item.needs_review:
                flagged += 1
                flag = " ⚠️ needs review"
            typer.secho(f"    ✓ {item.page_name}: {action}{flag}", fg=typer.colors.GREEN)

        for name in outcome.unmatched:
            still_failing += 1
            typer.secho(f"    ✗ {name}: not extracted on retry", fg=typer.colors.RED)

    cleared = sum(1 for outcome in outcomes if outcome.ok)
    typer.echo("")
    typer.echo(
        f"{len(outcomes)} source(s): {fixed} item(s) recovered, {still_failing} still failing, "
        f"{flagged} flagged, {cleared} source(s) now clean."
    )
    return 1 if still_failing else 0


@app.command()
def retry(
    source_ids: Optional[list[int]] = typer.Argument(
        None, help="Source ids to retry. Defaults to every failed source."
    ),
    namespace: Optional[str] = typer.Option(
        None, "--namespace", "-n", help="Only retry failed sources in this namespace."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logging."),
) -> None:
    """Re-run only the items a document failed on, leaving its good pages alone.

    For a document that mostly landed: `source.error_msg` names the items that
    failed, and this re-extracts the document, picks those items back out, and
    folds just them into their pages under the document's original source row.
    Successful pages are never rewritten, and citations keep their numbers.

    A document that failed outright — nothing extracted, no items — is not
    retried here; delete its source row and run `ingest` again.
    """
    _configure_logging(verbose)

    if not settings.openrouter_api_key:
        typer.secho(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    def run() -> int:
        conn = connect(settings.db_url)
        try:
            init_db(conn)

            if source_ids:
                sources = []
                for source_id in source_ids:
                    row = repo.get_source(conn, source_id)
                    if row is None:
                        typer.secho(f"No source #{source_id}.", fg=typer.colors.RED)
                        return 2
                    if row["status"] != "failed":
                        typer.secho(
                            f"Source #{source_id} ({row['filename']}) is not failed; skipping.",
                            fg=typer.colors.YELLOW,
                        )
                        continue
                    sources.append(row)
            else:
                sources = repo.list_failed_sources(conn, namespace)

            if not sources:
                typer.secho("Nothing to retry.", fg=typer.colors.YELLOW)
                return 0

            typer.echo(f"Retrying failed items in {len(sources)} source(s)\n")

            client = LLMClient()
            deps = Deps(conn=conn, client=client, settings=settings)
            outcomes = retry_sources(deps, sources)
            try:
                telemetry.record_retries(conn, outcomes)
            except Exception as exc:  # telemetry must never fail a retry
                log.warning("failed to record retry telemetry: %s", exc)
            _link_touched_pages(conn, client, outcomes)
            return _report_retries(outcomes)
        finally:
            conn.close()

    raise typer.Exit(code=run())


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
    conn = connect(settings.db_url)
    try:
        init_db(conn)
        namespaces = conn.execute(
            "SELECT namespace, COUNT(*) AS pages, "
            "COUNT(*) FILTER (WHERE needs_review) AS flagged "
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
    """Wipe the wiki pages and drop every database table, back to a clean slate.

    Raw source documents are never touched — re-running `ingest` rebuilds
    everything from them.
    """
    from llm_wiki.db.connection import DROP_ALL_SQL

    wiki_dir = settings.wiki_dir

    # Guard against a misconfigured LLM_WIKI_WIKI_DIR turning this into an
    # rm -rf of the project (or of the raw corpus we promise not to touch).
    for forbidden, label in ((settings.raw_dir, "raw_dir"), (PROJECT_ROOT, "project root")):
        if wiki_dir == forbidden:
            typer.secho(f"Refusing to reset: wiki_dir is the {label}.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    pages = sorted(p for p in wiki_dir.rglob("*") if p.is_file()) if wiki_dir.exists() else []

    typer.echo(f"This will delete {len(pages)} page(s) under {wiki_dir}")
    typer.echo(f"and drop every llm-wiki table in {settings.db_url}.")
    typer.secho(f"{settings.raw_dir} will not be touched.", fg=typer.colors.GREEN)
    if not yes:
        typer.confirm("Proceed?", abort=True)

    if wiki_dir.exists():
        shutil.rmtree(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(settings.db_url)
    try:
        conn.execute(DROP_ALL_SQL)
    finally:
        conn.close()

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

    if not settings.openrouter_api_key:
        typer.secho(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    def run() -> None:
        conn = connect(settings.db_url)
        try:
            init_db(conn)

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

            client = LLMClient()
            total_changed = 0
            for ns in namespaces:
                page_ids = [row["page_id"] for row in repo.list_pages(conn, ns)]
                results = links.link_pages(
                    conn,
                    wiki_dir=settings.wiki_dir,
                    namespace=ns,
                    page_ids=page_ids,
                    client=client,
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
                    typer.secho(
                        f"    · {r.page_name}: {len(r.links)} link(s)", fg=typer.colors.CYAN
                    )

            if dry_run and total_changed:
                typer.echo("\nDry run: no files were written.")
        finally:
            conn.close()

    run()


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
