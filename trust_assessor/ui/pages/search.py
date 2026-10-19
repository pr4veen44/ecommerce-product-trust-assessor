"""Product search with filters; results link to product detail and can be sent to Compare."""

from __future__ import annotations

from nicegui import ui

from trust_assessor.db import repository as repo
from trust_assessor.ui.components import card, heading, product_table, setup_required
from trust_assessor.ui.layout import frame

GRADES = ["High trust", "Moderate trust", "Low trust", "Insufficient evidence"]
SORTS = {"Trust score": "trust_score", "Number of reviews": "n_reviews", "Average rating": "mean_rating",
         "Evidence coverage": "evidence_coverage", "Title": "title"}


def register() -> None:
    @ui.page("/search")
    def search_page(q: str = "") -> None:
        with frame("Product Search", "search"):
            if repo.count_products() == 0:
                setup_required()
                return

            state = {"q": q, "grades": [], "range": {"min": 0, "max": 100}, "min_reviews": 0,
                     "sort": "Trust score", "desc": True, "limit": 200}

            with card():
                with ui.element("div").classes("grid grid-cols-1 md:grid-cols-6 gap-3 items-end w-full"):
                    search = ui.input("Search title, brand or product ID", value=q,
                                      on_change=lambda e: (state.update(q=e.value or ""), results.refresh())
                                      ).props("outlined dense clearable debounce=350").classes("md:col-span-3")
                    with search.add_slot("prepend"):
                        ui.icon("search")
                    ui.select(GRADES, multiple=True, label="Grade", value=[],
                              on_change=lambda e: (state.update(grades=e.value), results.refresh())
                              ).props("outlined dense use-chips").classes("md:col-span-3")
                    ui.number("Min reviews", value=0, min=0, step=5,
                              on_change=lambda e: (state.update(min_reviews=int(e.value or 0)), results.refresh())
                              ).props("outlined dense").classes("md:col-span-1")
                    ui.select(list(SORTS), label="Sort by", value="Trust score",
                              on_change=lambda e: (state.update(sort=e.value), results.refresh())
                              ).props("outlined dense").classes("md:col-span-2")
                    ui.switch("Descending", value=True,
                              on_change=lambda e: (state.update(desc=e.value), results.refresh())).classes("md:col-span-1")
                    with ui.column().classes("md:col-span-2 gap-0 w-full"):
                        ui.label("Trust score range").classes("text-xs text-slate-500")
                        ui.range(min=0, max=100, step=5, value={"min": 0, "max": 100},
                                 on_change=lambda e: (state.update(range=e.value), results.refresh())
                                 ).props("label-always dense")

            table_holder: dict = {}

            @ui.refreshable
            def results() -> None:
                lo, hi = state["range"]["min"], state["range"]["max"]
                rows, total = repo.search_products(
                    state["q"], grades=state["grades"] or None, min_score=lo if lo > 0 else None,
                    max_score=hi if hi < 100 else None, min_reviews=state["min_reviews"],
                    sort_by=SORTS[state["sort"]], descending=state["desc"], limit=state["limit"])
                with card():
                    with ui.row().classes("items-center justify-between w-full"):
                        heading(f"{total:,} products", f"Showing the first {len(rows):,}. Click a row to open it, or tick rows to compare."
                                if total > len(rows) else "Click a row to open it, or tick rows to compare.")
                        compare_btn = ui.button("Compare selected", icon="compare_arrows").props("outline color=primary")
                    if not rows:
                        ui.label("No products match these filters.").classes("tc-sub py-6")
                        compare_btn.disable()
                        return
                    table = product_table(rows, selection="multiple")
                    table_holder["t"] = table

                    def go_compare() -> None:
                        ids = [r["parent_asin"] for r in table.selected][:4]
                        if len(ids) < 2:
                            ui.notify("Select 2 to 4 products to compare.", type="warning")
                            return
                        ui.navigate.to("/compare?ids=" + ",".join(ids))

                    compare_btn.on_click(go_compare)

            results()
