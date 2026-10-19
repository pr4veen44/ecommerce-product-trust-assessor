"""Dashboard: overview of the assessed catalogue."""

from __future__ import annotations

from nicegui import ui

from trust_assessor.db import repository as repo
from trust_assessor.services import analytics
from trust_assessor.services.assessment_service import load_scoring_config
from trust_assessor.ui import charts, theme
from trust_assessor.ui.components import card, heading, info_note, kpi, product_table, setup_required
from trust_assessor.ui.layout import frame


def _chip(icon: str, text: str) -> None:
    with ui.row().classes("items-center gap-1 px-2 py-1 rounded-full bg-slate-100 no-wrap"):
        ui.icon(icon, size="16px").classes("text-slate-500")
        ui.label(text).classes("text-xs text-slate-700")


def _mini_list(rows: list[dict]) -> None:
    if not rows:
        ui.label("No products match.").classes("tc-sub")
        return
    for r in rows:
        with ui.link(target=f"/product/{r['parent_asin']}").classes("no-underline w-full"):
            with ui.row().classes("w-full items-center justify-between no-wrap py-1 hover:bg-slate-50 rounded px-1"):
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label(theme.short(r["title"], 70)).classes("text-sm font-medium text-slate-800")
                    ui.label(f"{r['n_reviews']:,} reviews · {r.get('grade') or 'n/a'}").classes("text-xs text-slate-500")
                ui.badge(theme.fmt_score(r["trust_score"]),
                         color=theme.GRADE_QUASAR.get(r.get("grade") or "", "grey")).classes("text-sm")


def register() -> None:
    @ui.page("/")
    def dashboard_page() -> None:
        with frame("Dashboard", "dashboard"):
            info = analytics.pipeline_info()
            if not info["has_data"] or repo.count_products() == 0:
                setup_required()
                return

            summ = analytics.dashboard_summary()
            cfg = load_scoring_config().to_dict()
            stats, ds, mm = info["stats"] or {}, info["dataset"] or {}, info["models"] or {}

            with ui.row().classes("gap-2 flex-wrap"):
                _chip("category", f"Category: {stats.get('n_products') and ds.get('category', 'n/a')}")
                _chip("date_range", f"Reviews {stats.get('date_min', '?')} → {stats.get('date_max', '?')}")
                _chip("sentiment_satisfied", f"Sentiment: {stats.get('sentiment_method', 'none')}")
                _chip("shield", f"Suspicious signals: {stats.get('suspicion_method', 'n/a').replace('_', ' ')}")
                _chip("model_training", f"Models: {mm.get('model_version', 'not trained')}")

            if not mm:
                info_note("ML models have not been trained yet, so suspicious-review detection uses rules only and the "
                          "informativeness component is unavailable. Run python scripts/train_models.py then "
                          "python scripts/assess_products.py.", "warning", "amber")

            flags = analytics.review_flag_stats()
            high = summ["grades"].get("High trust", 0)
            with ui.element("div").classes("grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 w-full"):
                kpi("Products", f"{summ['n_products']:,}", "inventory_2")
                kpi("Reviews analysed", f"{flags['n_reviews']:,}", "rate_review", color="#0ea5e9")
                kpi("Mean trust score", theme.fmt_score(summ["mean_score"]), "speed", f"median {theme.fmt_score(summ['median_score'])}")
                kpi("High-trust products", theme.fmt_pct(high / max(summ["n_products"], 1)), "verified", f"{high:,} products", "#16a34a")
                kpi("Reviews flagged", theme.fmt_pct(flags["share_flagged"], 1), "flag", f"{flags['n_flagged']:,} warrant a closer look", "#d97706")
                kpi("Avg evidence coverage", theme.fmt_pct(summ["mean_coverage"]), "fact_check", "share of score weight with data", "#8b5cf6")

            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-3 gap-4 w-full"):
                with card("lg:col-span-2"):
                    heading("Trust score distribution", "Dotted lines mark the grade thresholds.")
                    ui.plotly(charts.score_histogram(summ["scores"], cfg)).classes("w-full")
                with card():
                    heading("Grades")
                    ui.plotly(charts.grade_donut(summ["grades"])).classes("w-full")

            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-3 gap-4 w-full"):
                with card():
                    heading("Average component profile", "Mean score over products where the component has data.")
                    ui.plotly(charts.mean_component_profile(summ["component_means"])).classes("w-full")
                with card():
                    heading("Highest scores", "Products with at least 10 reviews.")
                    top, _ = repo.search_products(min_reviews=10, sort_by="trust_score", descending=True, limit=7)
                    _mini_list(top)
                with card():
                    heading("Lowest scores", "Products with at least 10 reviews.")
                    low, _ = repo.search_products(min_reviews=10, sort_by="trust_score", descending=False, limit=7)
                    _mini_list([r for r in low if r["trust_score"] is not None])

            with card():
                heading("Most-reviewed products")
                rows, _ = repo.search_products(sort_by="n_reviews", descending=True, limit=10)
                product_table(rows, page_size=10)
