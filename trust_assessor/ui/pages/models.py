"""Model Insights: real evaluation metrics and feature importance, or an explicit 'not available'."""

from __future__ import annotations

from nicegui import ui

from trust_assessor.db import repository as repo
from trust_assessor.features.suspicion_rules import RULE_CATALOG
from trust_assessor.ml import registry
from trust_assessor.services import analytics
from trust_assessor.ui import charts, theme
from trust_assessor.ui.components import card, heading, info_note, setup_required
from trust_assessor.ui.layout import frame


def _kv(rows: list[tuple[str, str]]) -> None:
    with ui.column().classes("gap-1 w-full"):
        for k, v in rows:
            with ui.row().classes("w-full justify-between no-wrap gap-4"):
                ui.label(k).classes("text-sm text-slate-500")
                ui.label(v).classes("text-sm font-medium text-right")


def register() -> None:
    @ui.page("/models")
    def models_page() -> None:
        with frame("Model Insights", "models"):
            info = analytics.pipeline_info()
            if not info["has_data"]:
                setup_required()
                return
            ds, mm, stats = info["dataset"] or {}, info["models"], info["stats"] or {}

            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-2 gap-4 w-full"):
                with card():
                    heading("Data & model versions", "Stored with every assessment for reproducibility")
                    _kv([("Dataset version", ds.get("dataset_version", "n/a")),
                         ("Source", ds.get("source", "n/a")),
                         ("Reviews / products", f"{stats.get('n_reviews', 0):,} / {stats.get('n_products', 0):,}"),
                         ("Model version", (mm or {}).get("model_version", "not trained")),
                         ("Trained at", (mm or {}).get("trained_at", "n/a")),
                         ("Random seed", str(ds.get("seed", "n/a")))])
                    if ds.get("sampling_note"):
                        ui.label(ds["sampling_note"]).classes("tc-sub mt-2")
                with card():
                    heading("Sentiment model")
                    rep = info["sentiment_report"]
                    if not rep:
                        info_note("Sentiment analysis was not run (SENTIMENT_BACKEND=none).", "info", "amber")
                    else:
                        _kv([("Method", rep["method"]), ("Reviews scored", f"{rep['n_scored']:,}")] +
                            [(f"Share {k}", theme.fmt_pct(v)) for k, v in rep["label_distribution"].items()])
                        if rep["method"].startswith("lexicon"):
                            info_note("Lexicon fallback in use: much weaker than DistilBERT. Re-run prepare_dataset.py with the transformers backend for better results.", "warning", "amber")
                        elif "distilbert" in rep["method"].lower():
                            ui.label("Pretrained DistilBERT (SST-2) is binary; 'neutral' is derived from low model confidence, not a trained class.").classes("tc-sub mt-1")
                        if "spearman_score_vs_rating" in rep:
                            ui.label("Agreement with star ratings (a noisy proxy, not accuracy):").classes("text-sm font-semibold mt-2")
                            _kv([("Spearman ρ (score vs rating)", f"{rep['spearman_score_vs_rating']:.3f}")] +
                                ([("ROC-AUC vs 4-5★ / 1-2★ proxy", f"{rep['roc_auc_vs_proxy']:.3f}")] if "roc_auc_vs_proxy" in rep else []) +
                                ([("Agreement on non-neutral reviews", theme.fmt_pct(rep["agreement_non_neutral"], 1))] if "agreement_non_neutral" in rep else []))
                        ui.label(rep["caveat"]).classes("tc-sub mt-1")

            # ------------------------------------------------------------ helpfulness model
            heading("Supervised model: review helpfulness (XGBoost + SHAP)",
                    "Target: helpful_vote ≥ 1, the only genuine observed label in the data. Not a measure of authenticity.")
            hm = registry.metrics(registry.HELPFULNESS_METRICS)
            shap_g = registry.metrics(registry.HELPFULNESS_SHAP)
            if not hm:
                info_note("The helpfulness model has not been trained (or there was too little labelled data), so there are no metrics to show. "
                          "Run python scripts/train_models.py.", "warning", "amber")
            else:
                x, b = hm["xgboost"], hm["baseline_logistic_regression"]
                with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-3 gap-4 w-full"):
                    with card("lg:col-span-2"):
                        heading("Held-out test performance", f"{hm['n_test']:,} reviews from {hm['n_test_products']:,} unseen products")
                        rows = [{"metric": "ROC-AUC", "xgb": f"{x['roc_auc']:.3f}", "base": f"{b['roc_auc']:.3f}"},
                                {"metric": "PR-AUC", "xgb": f"{x['pr_auc']:.3f}", "base": f"{b['pr_auc']:.3f}"},
                                {"metric": "F1 (validation-tuned threshold)", "xgb": f"{x['f1']:.3f}", "base": f"{b['f1']:.3f}"},
                                {"metric": "Precision", "xgb": f"{x['precision']:.3f}", "base": f"{b['precision']:.3f}"},
                                {"metric": "Recall", "xgb": f"{x['recall']:.3f}", "base": f"{b['recall']:.3f}"},
                                {"metric": "Accuracy", "xgb": f"{x['accuracy']:.3f}", "base": f"{b['accuracy']:.3f}"},
                                {"metric": "Brier score (lower is better)", "xgb": f"{x['brier']:.3f}", "base": "-"}]
                        ui.table(columns=[{"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                                          {"name": "xgb", "label": "XGBoost", "field": "xgb", "align": "right"},
                                          {"name": "base", "label": "Logistic regression baseline", "field": "base", "align": "right"}],
                                 rows=rows, row_key="metric").classes("w-full").props("flat dense")
                        ui.label(f"A classifier with no skill scores PR-AUC ≈ {hm['baseline_majority_pr_auc']:.3f} (positive rate) and ROC-AUC 0.5.").classes("tc-sub mt-2")
                        ui.label(hm["split"] + ". " + hm["threshold_note"]).classes("tc-sub")
                        ui.label(hm["caveat"]).classes("tc-sub")
                    with card():
                        heading("Training data")
                        _kv([("Train / val / test", f"{hm['n_train']:,} / {hm['n_validation']:,} / {hm['n_test']:,}"),
                             ("Helpful rate (train)", theme.fmt_pct(hm["positive_rate_train"], 1)),
                             ("Helpful rate (test)", theme.fmt_pct(hm["positive_rate_test"], 1)),
                             ("Boosting rounds", str(hm["best_iteration"])), ("Features", str(len(hm["features"]))),
                             ("Out-of-fold ROC-AUC", f"{hm['oof_roc_auc']:.3f}")])
                if shap_g:
                    with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-2 gap-4 w-full"):
                        with card():
                            heading("Feature importance: mean |SHAP|", f"Computed on {shap_g['n_rows_explained']:,} held-out reviews. {shap_g['note']}")
                            ui.plotly(charts.importance_bars(shap_g["labels"], shap_g["mean_abs_shap"], "Mean |SHAP value| (log-odds)")).classes("w-full")
                        with card():
                            heading("Feature importance: XGBoost gain")
                            ui.plotly(charts.importance_bars(shap_g["labels"], shap_g["gain_importance"], "Total gain", "#0ea5e9")).classes("w-full")
                    with card():
                        heading("Direction of effect", "Spearman correlation between a feature's value and its SHAP value (positive: higher values raise predicted helpfulness)")
                        pairs = sorted(zip(shap_g["labels"], shap_g["direction_corr"], shap_g["mean_abs_shap"]), key=lambda p: -p[2])
                        ui.table(columns=[{"name": "f", "label": "Feature", "field": "f", "align": "left"},
                                          {"name": "d", "label": "Direction (ρ)", "field": "d", "align": "right"},
                                          {"name": "m", "label": "Mean |SHAP|", "field": "m", "align": "right"}],
                                 rows=[{"f": l, "d": "n/a" if d is None else f"{d:+.2f}", "m": f"{m:.3f}"} for l, d, m in pairs],
                                 row_key="f", pagination={"rowsPerPage": 10}).classes("w-full").props("flat dense")

            # ------------------------------------------------------------ anomaly model + rules
            heading("Suspicious-review detection (unsupervised + rules)",
                    "No ground-truth labels exist, so no accuracy is reported and reviews are never declared fake.")
            ar = registry.metrics(registry.ANOMALY_REPORT)
            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-2 gap-4 w-full"):
                with card():
                    heading("Anomaly model (IsolationForest)")
                    if not ar:
                        info_note("Not trained: suspicion scores currently use the transparent rules only.", "warning", "amber")
                    else:
                        _kv([("Algorithm", ar["algorithm"]), ("Trees / contamination", f"{ar['n_estimators']} / {ar['contamination']}"),
                             ("Reviews fitted", f"{ar['n_reviews_fitted']:,}"),
                             ("Flag threshold (suspicion score)", f"{ar['flag_threshold']:.2f}"),
                             ("Reviews flagged (rules + model)", f"{ar['n_flagged_reviews']:,} ({theme.fmt_pct(ar['share_flagged_reviews'], 1)})"),
                             ("Flagged by rules alone", theme.fmt_pct(ar["share_flagged_by_rules_only"], 1))])
                        ui.label(ar["evaluation"]).classes("tc-sub mt-2")
                        ui.label("Features: " + ", ".join(ar["features"])).classes("tc-sub tc-mono")
                with card():
                    heading("Suspicion score distribution")
                    fl = analytics.review_flag_stats()
                    ui.plotly(charts.suspicion_hist(fl["suspicion_scores"], stats.get("suspicion_flag_threshold", 0.5))).classes("w-full")
            with card():
                heading("Rule catalogue", "Weights add up per review (capped at 1.0) before the anomaly model is blended in.")
                rates = analytics.dataset_rule_rates()
                ui.table(columns=[{"name": "r", "label": "Signal", "field": "r", "align": "left"},
                                  {"name": "w", "label": "Weight", "field": "w", "align": "right"},
                                  {"name": "p", "label": "Share of reviews", "field": "p", "align": "right"},
                                  {"name": "d", "label": "Definition", "field": "d", "align": "left", "style": "white-space:normal"}],
                         rows=[{"r": m["label"], "w": f"{m['weight']:.2f}", "p": theme.fmt_pct(rates.get(m["label"]), 1), "d": m["description"]}
                               for m in RULE_CATALOG.values()], row_key="r").classes("w-full").props("flat dense")

            with card():
                heading("Registered versions (SQLite)")
                v = repo.version_rows()
                ui.table(columns=[{"name": "v", "label": "Dataset version", "field": "v", "align": "left"},
                                  {"name": "n", "label": "Reviews", "field": "n", "align": "right"},
                                  {"name": "p", "label": "Products", "field": "p", "align": "right"}],
                         rows=[{"v": d["version"], "n": f"{d['n_reviews']:,}", "p": f"{d['n_products']:,}"} for d in v["datasets"]],
                         row_key="v").classes("w-full").props("flat dense")
