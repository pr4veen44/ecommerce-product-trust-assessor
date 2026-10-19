"""Product detail page."""

from __future__ import annotations

import pandas as pd
from nicegui import ui

from trust_assessor.data.storage import DataStore
from trust_assessor.db import repository as repo
from trust_assessor.explain import shap_explainer
from trust_assessor.explain.score_explainer import decomposition, explain_text
from trust_assessor.features.product_features import COMPLETENESS_FIELDS, parse_json_dict, parse_json_list
from trust_assessor.features.suspicion_rules import triggered_rules
from trust_assessor.services import analytics
from trust_assessor.services.assessment_service import assess_one
from trust_assessor.ui import charts, theme
from trust_assessor.ui.components import (bullet_list, card, clean_value, confidence_chip, grade_badge, heading,
                                          info_note, product_thumb)
from trust_assessor.ui.layout import frame

REVIEW_COLUMNS = [
    {"name": "date", "label": "Date", "field": "date", "align": "left", "sortable": True},
    {"name": "rating", "label": "★", "field": "rating", "align": "right", "sortable": True},
    {"name": "sentiment", "label": "Sentiment", "field": "sentiment", "align": "left"},
    {"name": "verified", "label": "Verified", "field": "verified", "align": "left"},
    {"name": "helpful", "label": "Helpful votes", "field": "helpful", "align": "right", "sortable": True},
    {"name": "suspicion", "label": "Suspicion", "field": "suspicion", "align": "right", "sortable": True},
    {"name": "snippet", "label": "Review", "field": "snippet", "align": "left"},
]
SUSP_CELL = """<q-td :props="props"><q-badge :color="props.row.flag ? 'negative' : 'grey-5'" :label="props.row.suspicion.toFixed(2)" /></q-td>"""
SNIPPET_CELL = """<q-td :props="props" style="max-width:520px;white-space:normal"><div class="tc-clamp2">{{ props.row.snippet }}</div></q-td>"""


def _review_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for r in df.itertuples():
        rows.append({
            "review_id": r.review_id, "date": r.timestamp.strftime("%Y-%m-%d"), "rating": int(r.rating),
            "sentiment": clean_value(r.sent_label) or "n/a", "verified": "Yes" if r.verified_purchase else "No",
            "helpful": int(r.helpful_vote), "suspicion": float(r.suspicion_score), "flag": bool(r.suspicion_flag),
            "snippet": theme.short((r.title + " — " if r.title else "") + r.text, 240),
        })
    return rows


def _review_table(df: pd.DataFrame, on_open) -> None:
    if df.empty:
        ui.label("No reviews match.").classes("tc-sub py-4")
        return
    t = ui.table(columns=REVIEW_COLUMNS, rows=_review_rows(df), row_key="review_id",
                 pagination={"rowsPerPage": 10}).classes("w-full").props("flat dense")
    t.add_slot("body-cell-suspicion", SUSP_CELL)
    t.add_slot("body-cell-snippet", SNIPPET_CELL)
    t.on("row-click", lambda e: on_open(e.args[1]["review_id"]))


def _open_review(review_id: str) -> None:
    row = DataStore().review_row(review_id)
    if row is None:
        ui.notify("Review not found", type="warning")
        return
    with ui.dialog().props("maximized=false") as dlg, ui.card().classes("w-[min(920px,96vw)] max-h-[90vh] overflow-auto gap-3"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(row["title"] or "(no title)").classes("tc-h2")
            ui.button(icon="close", on_click=dlg.close).props("flat round dense")
        with ui.row().classes("gap-2 flex-wrap"):
            ui.badge(f"{int(row['rating'])}★", color="primary")
            ui.badge("Verified" if row["verified_purchase"] else "Unverified", color="positive" if row["verified_purchase"] else "grey")
            ui.badge(f"{int(row['helpful_vote'])} helpful votes", color="secondary")
            ui.badge(f"Sentiment: {clean_value(row.get('sent_label')) or 'n/a'}", color="info")
            ui.badge(f"Suspicion {row['suspicion_score']:.2f}", color="negative" if row["suspicion_flag"] else "grey")
        ui.label(row["text"] or "(no text)").classes("text-sm text-slate-800 whitespace-pre-wrap")
        ui.separator()
        heading("Why was / wasn't this review flagged?", "Rule-based reasons (transparent) and SHAP for the anomaly model. A flag means 'look closer', never 'fake'.")
        rules = triggered_rules(row)
        if rules:
            bullet_list([f"{r['label']}: {r['description']}" for r in rules], "flag", "#dc2626")
        else:
            ui.label("No rule-based signals triggered.").classes("tc-sub")
        ano = shap_explainer.explain_anomaly(row)
        if ano:
            ui.label("SHAP: features driving the anomaly model's score (red = pushes toward 'more unusual')").classes("text-sm font-medium mt-2")
            ui.plotly(charts.shap_local_bars(ano["items"], label_key="feature", flip=True, height=300)).classes("w-full")
        hel = shap_explainer.explain_helpfulness(row)
        if hel:
            ui.separator()
            heading("Predicted helpfulness (XGBoost + SHAP)",
                    f"Model estimates a {hel['prediction']:.0%} chance of receiving a helpful vote. Green bars raise it, red lower it ({hel['unit']}).")
            ui.plotly(charts.shap_local_bars(hel["items"], height=340)).classes("w-full")
        else:
            ui.label("Helpfulness model not trained: no SHAP explanation available.").classes("tc-sub")
    dlg.open()


def register() -> None:
    @ui.page("/product/{asin}")
    def product_page(asin: str) -> None:
        with frame("Product details", "search"):
            store = DataStore()
            prod = store.product(asin) if store.has_processed_data() else None
            if prod is None:
                with card():
                    ui.label("Product not found").classes("tc-h1")
                    ui.link("Back to search", "/search")
                return
            a = repo.latest_assessment(asin)
            if a is None:
                assess_one(asin)
                a = repo.latest_assessment(asin)
            reviews = store.reviews_for(asin)
            comps = a["components"]
            cfg = a["scoring_config"]

            # ---------------------------------------------------------------- header
            with card():
                with ui.row().classes("w-full gap-4 no-wrap items-start"):
                    product_thumb(clean_value(prod.get("image_url")), "w-24 h-24 flex-none")
                    with ui.column().classes("gap-1 min-w-0 grow"):
                        ui.label(prod["title"]).classes("tc-h1 tc-clamp2")
                        bits = [f"ID {asin}"]
                        if clean_value(prod.get("store")):
                            bits.append(str(prod["store"]))
                        bits.append(str(prod.get("category") or "Unknown category"))
                        bits.append(f"Price {theme.fmt_price(clean_value(prod.get('price')))}")
                        ui.label(" · ".join(bits)).classes("tc-sub")
                        with ui.row().classes("items-center gap-3 mt-1 flex-wrap"):
                            grade_badge(a["grade"])
                            confidence_chip(a["confidence"])
                            ui.label(f"Evidence coverage {a['evidence_coverage']:.0%}").classes("text-sm text-slate-600")
                            ui.label(f"{int(prod['n_reviews']):,} reviews analysed").classes("text-sm text-slate-600")
                    with ui.column().classes("gap-2 flex-none"):
                        def reassess() -> None:
                            assess_one(asin)
                            analytics.clear_cache()
                            ui.notify("Re-assessed with the current scoring configuration.", type="positive")
                            ui.navigate.reload()
                        ui.button("Re-assess", icon="refresh", on_click=reassess).props("outline color=primary dense")
                        ui.button("Compare", icon="compare_arrows",
                                  on_click=lambda: ui.navigate.to(f"/compare?ids={asin}")).props("outline color=primary dense")
                ui.label(f"Assessed {a['created_at']:%Y-%m-%d %H:%M} UTC · dataset {a['dataset_version']} · models {a['model_version'] or 'untrained'}"
                         f" · sentiment {a['sentiment_method'] or 'none'} · config {a['config_hash']}").classes("tc-sub tc-mono mt-1")

            # ---------------------------------------------------------------- score + breakdown
            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-5 gap-4 w-full"):
                with card("lg:col-span-2"):
                    heading("Product Trust Score", "0-100 weighted combination of the components")
                    ui.plotly(charts.gauge(a["trust_score"], cfg)).classes("w-full")
                    if a["adjustments"]:
                        info_note(" ".join(a["adjustments"]), "gpp_maybe", "red")
                    if a["trust_score"] is None:
                        info_note("Not enough evidence to produce a score. See warnings below.", "help", "amber")
                with card("lg:col-span-3"):
                    heading("Component breakdown", "Each bar is a normalised 0-100 component. Hover for weights.")
                    ui.plotly(charts.component_bars(comps)).classes("w-full")

            with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 w-full"):
                with card():
                    heading("Strengths")
                    bullet_list(a["strengths"], "check_circle", "#16a34a", "No component stands out as strong.")
                with card():
                    heading("Warnings")
                    bullet_list(a["warnings"], "warning", "#d97706", "No warnings raised.")
                with card():
                    heading("Missing information", "Excluded from the score, never imputed.")
                    bullet_list(a["missing"], "help_outline", "#64748b", "All expected data was available.")
                with card():
                    heading("Not assessed", "The dataset cannot support these.")
                    bullet_list([f"{n['item']}: {n['reason']}" for n in _not_assessed()], "block", "#94a3b8")

            # ---------------------------------------------------------------- tabs
            with ui.tabs().classes("w-full bg-white rounded-lg border") as tabs:
                t_expl = ui.tab("Score explanation", icon="calculate")
                t_rev = ui.tab("Review analytics", icon="analytics")
                t_susp = ui.tab("Suspicious signals", icon="flag")
                t_list = ui.tab("Listing quality", icon="fact_check")
                t_hist = ui.tab("History", icon="history")
            with ui.tab_panels(tabs, value=t_expl).classes("w-full bg-transparent").props("animated=false"):
                with ui.tab_panel(t_expl).classes("p-0"):
                    _explanation_tab(a)
                with ui.tab_panel(t_rev).classes("p-0"):
                    _reviews_tab(prod, reviews)
                with ui.tab_panel(t_susp).classes("p-0"):
                    _suspicious_tab(a, reviews)
                with ui.tab_panel(t_list).classes("p-0"):
                    _listing_tab(prod)
                with ui.tab_panel(t_hist).classes("p-0"):
                    _history_tab(asin)


def _not_assessed() -> list[dict]:
    from trust_assessor.scoring.engine import NOT_ASSESSED

    return [{"item": i, "reason": r} for i, r in NOT_ASSESSED]


def _explanation_tab(a: dict) -> None:
    with ui.column().classes("gap-4 w-full"):
        with card():
            heading("How this score was calculated", "An exact decomposition: contributions add up to the final number.")
            with ui.column().classes("gap-1 mt-2"):
                for line in explain_text(a):
                    ui.label(line).classes("text-sm text-slate-800")
        with card():
            heading("Contribution waterfall")
            ui.plotly(charts.waterfall(a)).classes("w-full")
        with card():
            heading("Component details")
            rows = [{"label": d["label"], "score": "n/a" if d["score"] is None else f"{d['score']:.0f}",
                     "weight": f"{d['effective_weight']:.1%}" if d["available"] else f"({d['configured_weight']:.0%}) excluded",
                     "points": f"{d['contribution']:.1f}", "evidence": d["evidence"], "what": d["description"]}
                    for d in decomposition(a)]
            ui.table(columns=[
                {"name": "label", "label": "Component", "field": "label", "align": "left"},
                {"name": "score", "label": "Score", "field": "score", "align": "right"},
                {"name": "weight", "label": "Effective weight", "field": "weight", "align": "right"},
                {"name": "points", "label": "Points", "field": "points", "align": "right"},
                {"name": "evidence", "label": "Evidence", "field": "evidence", "align": "left", "style": "white-space:normal;min-width:320px"},
            ], rows=rows, row_key="label").classes("w-full").props("flat dense")
            ui.label("Scoring is a transparent heuristic, not validated against ground truth. See Methodology.").classes("tc-sub mt-2")


def _reviews_tab(prod: pd.Series, reviews: pd.DataFrame) -> None:
    with ui.column().classes("gap-4 w-full"):
        with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-3 gap-4 w-full"):
            with card():
                heading("Rating distribution")
                ui.plotly(charts.rating_distribution(prod)).classes("w-full")
            with card():
                heading("Sentiment", "Share of scored reviews" + (" (neutral = model uncertain)" if True else ""))
                ui.plotly(charts.sentiment_donut(prod)).classes("w-full")
            with card():
                heading("Key figures")
                facts = [("Mean rating (sample)", theme.fmt_score(prod["mean_rating"], 2)),
                         ("Catalogue rating", theme.fmt_score(clean_value(prod.get("meta_avg_rating")), 2)),
                         ("Catalogue rating count", theme.fmt_int(clean_value(prod.get("meta_rating_number")))),
                         ("Verified purchases", theme.fmt_pct(prod["verified_share"])),
                         ("5★ share", theme.fmt_pct(prod["share_5star"])), ("1★ share", theme.fmt_pct(prod["share_1star"])),
                         ("Sentiment scored", theme.fmt_int(prod["n_sentiment"])),
                         ("First / last review", f"{prod['first_review']:%Y-%m} → {prod['last_review']:%Y-%m}")]
                with ui.column().classes("gap-1 w-full"):
                    for k, v in facts:
                        with ui.row().classes("w-full justify-between no-wrap"):
                            ui.label(k).classes("text-sm text-slate-500")
                            ui.label(v).classes("text-sm font-medium")
        with card():
            heading("Ratings over time")
            ui.plotly(charts.rating_timeline(reviews)).classes("w-full")
        with card():
            heading("Review explorer", "Click a review for its rule-based and SHAP explanations.")
            f = {"rating": "All", "sent": "All", "flag": False}

            @ui.refreshable
            def table() -> None:
                d = reviews
                if f["rating"] != "All":
                    d = d[d["rating"] == int(f["rating"][0])]
                if f["sent"] != "All":
                    d = d[d["sent_label"] == f["sent"].lower()]
                if f["flag"]:
                    d = d[d["suspicion_flag"]]
                _review_table(d.sort_values("timestamp", ascending=False).head(1000), _open_review)

            with ui.row().classes("gap-3 items-center"):
                ui.select(["All", "1★", "2★", "3★", "4★", "5★"], value="All", label="Rating",
                          on_change=lambda e: (f.update(rating=e.value), table.refresh())).props("outlined dense").classes("w-28")
                ui.select(["All", "Positive", "Neutral", "Negative"], value="All", label="Sentiment",
                          on_change=lambda e: (f.update(sent=e.value), table.refresh())).props("outlined dense").classes("w-36")
                ui.switch("Flagged only", on_change=lambda e: (f.update(flag=e.value), table.refresh()))
            table()


def _suspicious_tab(a: dict, reviews: pd.DataFrame) -> None:
    with ui.column().classes("gap-4 w-full"):
        info_note("These signals mark reviews that deserve a closer look. Amazon Reviews 2023 has no ground-truth fake-review "
                  "labels, so nothing here proves a review is fake, and genuine reviews can trip rules by chance.", "info", "blue")
        with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-2 gap-4 w-full"):
            with card():
                heading("Signal prevalence", "Share of this product's reviews triggering each rule vs the whole sample")
                ui.plotly(charts.rule_prevalence(analytics.product_rule_rates(reviews), analytics.dataset_rule_rates())).classes("w-full")
            with card():
                heading("Summary")
                n_flag = int(reviews["suspicion_flag"].sum())
                auth = next((c for c in a["components"] if c["key"] == "authenticity"), None)
                facts = [("Flagged reviews", f"{n_flag} of {len(reviews)} ({n_flag / max(len(reviews), 1):.0%})"),
                         ("Method", (auth or {}).get("details", {}).get("method", "n/a").replace("_", " ")),
                         ("Authenticity component", theme.fmt_score((auth or {}).get("score"), 0) + " / 100")]
                with ui.column().classes("gap-1 w-full"):
                    for k, v in facts:
                        with ui.row().classes("w-full justify-between no-wrap"):
                            ui.label(k).classes("text-sm text-slate-500")
                            ui.label(v).classes("text-sm font-medium")
        with card():
            heading("Highest-suspicion reviews", "Sorted by suspicion score. Click for the reasons.")
            _review_table(reviews.sort_values("suspicion_score", ascending=False).head(50), _open_review)


def _listing_tab(prod: pd.Series) -> None:
    with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-3 gap-4 w-full"):
        with card():
            heading("Listing completeness")
            if not prod["has_metadata"]:
                info_note("No catalogue metadata was found for this product in the sampled metadata, so listing completeness is not scored.", "help", "amber")
            else:
                for key, label in COMPLETENESS_FIELDS.items():
                    ok = bool(prod[key])
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("check_circle" if ok else "cancel", size="20px").style(f"color:{'#16a34a' if ok else '#dc2626'}")
                        ui.label(label).classes("text-sm")
        with card("lg:col-span-2"):
            heading("Catalogue content")
            desc = clean_value(prod.get("description"))
            feats = parse_json_list(prod.get("features"))
            details = parse_json_dict(prod.get("details"))
            cats = parse_json_list(prod.get("categories"))
            if cats:
                ui.label(" › ".join(map(str, cats))).classes("tc-sub")
            if feats:
                ui.label("Features").classes("text-sm font-semibold mt-2")
                bullet_list([str(x) for x in feats[:8]], "chevron_right", "#6366f1")
            if desc:
                ui.label("Description").classes("text-sm font-semibold mt-2")
                ui.label(theme.short(desc, 900)).classes("text-sm text-slate-700")
            if details:
                ui.label("Details").classes("text-sm font-semibold mt-2")
                ui.table(columns=[{"name": "k", "label": "Field", "field": "k", "align": "left"},
                                  {"name": "v", "label": "Value", "field": "v", "align": "left"}],
                         rows=[{"k": k, "v": v} for k, v in details.items()], row_key="k").classes("w-full").props("flat dense")
            if not (cats or feats or desc or details):
                ui.label("No descriptive content available.").classes("tc-sub")


def _history_tab(asin: str) -> None:
    rows = repo.assessment_history(asin, limit=200)
    with ui.column().classes("gap-4 w-full"):
        with card():
            heading("Score history for this product", "Each (re)assessment is stored with its configuration and data versions.")
            ui.plotly(charts.score_history(rows)).classes("w-full")
        with card():
            ui.table(columns=[
                {"name": "when", "label": "When (UTC)", "field": "when", "align": "left"},
                {"name": "score", "label": "Score", "field": "score", "align": "right"},
                {"name": "grade", "label": "Grade", "field": "grade", "align": "left"},
                {"name": "source", "label": "Source", "field": "source", "align": "left"},
                {"name": "cfg", "label": "Config", "field": "cfg", "align": "left"},
                {"name": "model", "label": "Model version", "field": "model", "align": "left"},
            ], rows=[{"id": r["id"], "when": f"{r['created_at']:%Y-%m-%d %H:%M}", "score": theme.fmt_score(r["trust_score"]),
                      "grade": r["grade"], "source": r["source"], "cfg": r["config_hash"], "model": r["model_version"] or "-"}
                     for r in rows], row_key="id", pagination={"rowsPerPage": 8}).classes("w-full").props("flat dense")
