"""Assessment history: every (re)assessment is stored with its config and data versions."""

from __future__ import annotations

from nicegui import ui

from trust_assessor.db import repository as repo
from trust_assessor.ui import charts, theme
from trust_assessor.ui.components import card, heading, setup_required
from trust_assessor.ui.layout import frame


def register() -> None:
    @ui.page("/history")
    def history_page() -> None:
        with frame("Assessment History", "history"):
            if repo.count_assessments() == 0:
                setup_required()
                return
            state = {"asin": "", "source": "All"}

            runs = repo.run_summaries(30)
            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-5 gap-4 w-full"):
                with card("lg:col-span-2"):
                    heading("Assessment runs", "Mean score of each batch / re-weighting run")
                    ui.plotly(charts.run_trend(runs)).classes("w-full")
                with card("lg:col-span-3"):
                    heading("Runs")
                    ui.table(columns=[
                        {"name": "started", "label": "Started (UTC)", "field": "started", "align": "left"},
                        {"name": "source", "label": "Source", "field": "source", "align": "left"},
                        {"name": "n", "label": "Products", "field": "n", "align": "right"},
                        {"name": "mean", "label": "Mean score", "field": "mean", "align": "right"},
                        {"name": "cfg", "label": "Config", "field": "cfg", "align": "left"},
                        {"name": "model", "label": "Model", "field": "model", "align": "left"},
                    ], rows=[{"id": r["run_id"], "started": f"{r['started']:%Y-%m-%d %H:%M}", "source": r["source"], "n": r["n"],
                              "mean": theme.fmt_score(r["mean_score"]), "cfg": r["config_hash"], "model": r["model_version"] or "-"}
                             for r in runs], row_key="id", pagination={"rowsPerPage": 6}).classes("w-full").props("flat dense")

            with card():
                heading("All assessments")
                with ui.row().classes("gap-3 items-center"):
                    ui.input("Product ID", on_change=lambda e: (state.update(asin=(e.value or "").strip()), listing.refresh())
                             ).props("outlined dense clearable debounce=400").classes("w-56")
                    ui.select(["All", "batch", "interactive", "reweight"], value="All", label="Source",
                              on_change=lambda e: (state.update(source=e.value), listing.refresh())).props("outlined dense").classes("w-40")

                @ui.refreshable
                def listing() -> None:
                    rows = repo.assessment_history(state["asin"] or None, None if state["source"] == "All" else state["source"], 500)
                    if state["asin"] and rows:
                        ui.plotly(charts.score_history(rows)).classes("w-full")
                    t = ui.table(columns=[
                        {"name": "when", "label": "When (UTC)", "field": "when", "align": "left", "sortable": True},
                        {"name": "title", "label": "Product", "field": "title", "align": "left"},
                        {"name": "score", "label": "Score", "field": "score", "align": "right", "sortable": True},
                        {"name": "grade", "label": "Grade", "field": "grade", "align": "left"},
                        {"name": "conf", "label": "Confidence", "field": "conf", "align": "left"},
                        {"name": "source", "label": "Source", "field": "source", "align": "left"},
                        {"name": "cfg", "label": "Config", "field": "cfg", "align": "left"},
                    ], rows=[{"id": r["id"], "asin": r["parent_asin"], "when": f"{r['created_at']:%Y-%m-%d %H:%M}",
                              "title": theme.short(r["title"], 70), "score": theme.fmt_score(r["trust_score"]), "grade": r["grade"],
                              "conf": r["confidence"], "source": r["source"], "cfg": r["config_hash"]} for r in rows],
                        row_key="id", pagination={"rowsPerPage": 12}).classes("w-full").props("flat dense")
                    t.on("row-click", lambda e: ui.navigate.to(f"/product/{e.args[1]['asin']}"))

                listing()
