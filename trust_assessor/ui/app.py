"""NiceGUI application assembly."""

from __future__ import annotations

import logging
import re

from nicegui import app, ui
from starlette.responses import Response

from trust_assessor.config import get_settings
from trust_assessor.logging_setup import setup_logging

log = logging.getLogger(__name__)
FAVICON_LINK = re.compile(rb'\s*<link href="[^"]*" rel="shortcut icon" />')


def build() -> None:
    from trust_assessor.ui.pages import compare, dashboard, history, methodology, models, product, search

    for page in (dashboard, search, product, compare, history, models, methodology):
        page.register()


def _startup() -> None:
    from trust_assessor.services.assessment_service import ensure_database_ready

    status = ensure_database_ready()
    log.info("Database status: %s", status)
    if status == "no_data":
        log.warning("No processed data found. Run scripts/prepare_dataset.py, train_models.py and assess_products.py")


def run() -> None:
    s = get_settings()
    setup_logging(s.log_level)
    build()
    app.on_startup(_startup)
    @app.middleware("http")
    async def remove_favicon(request, call_next):
        if request.url.path.endswith("/favicon.ico"):
            return Response(status_code=404)
        response = await call_next(request)
        if response.headers.get("x-nicegui-content") != "page":
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        headers = {key: value for key, value in response.headers.items() if key != "content-length"}
        return Response(FAVICON_LINK.sub(b"", body), status_code=response.status_code, headers=headers)

    ui.run(host=s.ui_host, port=s.ui_port, title="Product Trust Assessor", reload=False,
           show=False, storage_secret=s.storage_secret, dark=False)
