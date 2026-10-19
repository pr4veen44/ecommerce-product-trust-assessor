"""Side-by-side comparison of 2-4 products (selection kept in the URL: /compare?ids=A,B,C)."""

from __future__ import annotations

from nicegui import ui

from trust_assessor.data.storage import DataStore
from trust_assessor.db import repository as repo
from trust_assessor.ui import charts, theme
from trust_assessor.ui.components import bullet_list, card, clean_value, grade_badge, heading, info_note, setup_required
from trust_assessor.ui.layout import frame

MAX_ITEMS = 4


def _url(ids: list[str]) -> str:
    return "/compare" + (f"?ids={','.join(ids)}" if ids else "")


def register() -> None:
    @ui.page("/compare")
    def compare_page(ids: str = "") -> None:
        with frame("Compare products", "compare"):
            if repo.count_products() == 0:
                setup_required()
                return
            selected = list(dict.fromkeys(i for i in ids.split(",") if i))[:MAX_ITEMS]
            store = DataStore()
            products = {i: repo.get_product(i) for i in selected}
            selected = [i for i in selected if products[i]]

            with card():
                heading("Products to compare", f"Pick 2 to {MAX_ITEMS} products. The selection is kept in the page URL so you can share it.")
                with ui.row().classes("gap-2 flex-wrap mt-1"):
                    for i in selected:
                        ui.chip(theme.short(products[i]["title"], 40), icon="inventory_2", removable=True,
                                on_value_change=lambda e, i=i: ui.navigate.to(_url([x for x in selected if x != i]))
                                if not e.value else None).props("color=indigo-1 text-color=indigo-9")
                if len(selected) < MAX_ITEMS:
                    results = ui.column().classes("w-full gap-1")

                    def search(e) -> None:
                        results.clear()
                        q = (e.value or "").strip()
                        if len(q) < 2:
                            return
                        rows, _ = repo.search_products(q, limit=8, min_reviews=1)
                        with results:
                            for r in rows:
                                if r["parent_asin"] in selected:
                                    continue
                                ui.button(f"{theme.short(r['title'], 80)}  ·  {theme.fmt_score(r['trust_score'])}", icon="add",
                                          on_click=lambda r=r: ui.navigate.to(_url(selected + [r["parent_asin"]]))
                                          ).props("flat dense no-caps align=left").classes("w-full justify-start")

                    ui.input("Search a product to add", on_change=search).props("outlined dense clearable debounce=350").classes("w-full md:w-96 mt-2")

            if len(selected) < 2:
                info_note("Add at least two products to see the comparison.", "compare_arrows", "blue")
                return

            assess = {i: repo.latest_assessment(i) for i in selected}
            prods = {i: store.product(i) for i in selected}
            names = {i: theme.short(products[i]["title"], 28) for i in selected}
            series = {names[i]: assess[i]["components"] for i in selected}

            with ui.element("div").classes("grid grid-cols-1 xl:grid-cols-2 gap-4 w-full"):
                with card():
                    heading("Component profile")
                    ui.plotly(charts.radar(series)).classes("w-full")
                with card():
                    heading("Component scores")
                    ui.plotly(charts.grouped_component_bars(series)).classes("w-full")

            def v(i: str, fn) -> str:
                try:
                    return fn(i)
                except (KeyError, TypeError, ValueError):
                    return "n/a"

            metrics = [
                ("Trust score", lambda i: theme.fmt_score(assess[i]["trust_score"])),
                ("Grade", lambda i: assess[i]["grade"]),
                ("Confidence", lambda i: assess[i]["confidence"]),
                ("Evidence coverage", lambda i: f"{assess[i]['evidence_coverage']:.0%}"),
                ("Reviews analysed", lambda i: theme.fmt_int(prods[i]["n_reviews"])),
                ("Mean rating (sample)", lambda i: theme.fmt_score(prods[i]["mean_rating"], 2)),
                ("Catalogue rating", lambda i: theme.fmt_score(clean_value(prods[i]["meta_avg_rating"]), 2)),
                ("Catalogue rating count", lambda i: theme.fmt_int(clean_value(prods[i]["meta_rating_number"]))),
                ("Verified purchases", lambda i: theme.fmt_pct(prods[i]["verified_share"])),
                ("Positive sentiment", lambda i: theme.fmt_pct(clean_value(prods[i]["sent_pos_share"]))),
                ("Negative sentiment", lambda i: theme.fmt_pct(clean_value(prods[i]["sent_neg_share"]))),
                ("Reviews with suspicious signals", lambda i: theme.fmt_pct(prods[i]["suspicious_share"])),
                ("Price", lambda i: theme.fmt_price(clean_value(prods[i]["price"]))),
                ("Store", lambda i: clean_value(prods[i]["store"]) or "not listed"),
            ]
            rows = [{"metric": m, **{i: v(i, f) for i in selected}} for m, f in metrics]
            with card():
                heading("Side by side")
                cols = [{"name": "metric", "label": "", "field": "metric", "align": "left", "classes": "text-weight-medium"}]
                cols += [{"name": i, "label": names[i], "field": i, "align": "left"} for i in selected]
                ui.table(columns=cols, rows=rows, row_key="metric").classes("w-full").props("flat dense")
                with ui.row().classes("gap-2 mt-2"):
                    for i in selected:
                        ui.button(f"Open {names[i]}", on_click=lambda i=i: ui.navigate.to(f"/product/{i}")).props("flat dense no-caps color=primary")

            with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 w-full"):
                for i in selected:
                    with card():
                        ui.label(names[i]).classes("tc-h2")
                        grade_badge(assess[i]["grade"])
                        ui.label("Strengths").classes("text-sm font-semibold mt-2")
                        bullet_list(assess[i]["strengths"][:3], "check_circle", "#16a34a", "None highlighted.")
                        ui.label("Warnings").classes("text-sm font-semibold mt-2")
                        bullet_list(assess[i]["warnings"][:3], "warning", "#d97706", "None raised.")
