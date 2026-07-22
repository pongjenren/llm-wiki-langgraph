"""Starlette app serving the read-only ingest dashboard.

The dashboard never writes and never touches the vector index, so it only
ensures the core relational tables exist (cheap, no embedding model load) rather
than calling the full init_db.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from llm_wiki.config import Settings, settings
from llm_wiki.db.connection import SCHEMA_PATH, Connection, connect
from llm_wiki.dashboard.status import compute_raw_status

TEMPLATES_DIR = Path(__file__).with_name("templates")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def _format_dt(value: datetime | str | None) -> str:
    if not value:
        return "—"
    # TIMESTAMPTZ columns come back as datetime; older callers may pass ISO text.
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["duration"] = _format_duration
    templates.env.filters["dt"] = _format_dt
    return templates


def create_app(app_settings: Settings = settings) -> Starlette:
    templates = _templates()

    def _connect() -> Connection:
        conn = connect(app_settings.db_url)
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        return conn

    async def overview(request: Request) -> Response:
        conn = _connect()
        try:
            from llm_wiki.db import repo

            runs = repo.list_runs(conn, limit=50)
            latest = runs[0] if runs else None
            status = compute_raw_status(conn, app_settings.raw_dir)
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "overview.html",
            {
                "latest": latest,
                "runs": runs,
                "status": status,
                "raw_dir": str(app_settings.raw_dir),
            },
        )

    async def run_detail(request: Request) -> Response:
        run_id = int(request.path_params["run_id"])
        conn = _connect()
        try:
            from llm_wiki.db import repo

            run = repo.get_run(conn, run_id)
            docs = repo.list_run_docs(conn, run_id) if run else []
        finally:
            conn.close()
        if run is None:
            return HTMLResponse(f"Run {run_id} not found.", status_code=404)
        # items_json is a JSONB column, so psycopg2 returns it already parsed.
        docs_view = [{"row": doc, "entries": doc["items_json"] or []} for doc in docs]
        return templates.TemplateResponse(
            request, "run.html", {"run": run, "docs": docs_view}
        )

    return Starlette(
        routes=[
            Route("/", overview),
            Route("/runs/{run_id:int}", run_detail),
        ]
    )


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
