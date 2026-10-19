"""Methodology, limitations and the scoring-configuration editor."""

from __future__ import annotations

from nicegui import run, ui

from trust_assessor.config import get_settings
from trust_assessor.scoring.config import (COMPONENT_DESCRIPTIONS, COMPONENT_KEYS, COMPONENT_LABELS, DEFAULT_WEIGHTS,
                                           ScoringConfig)
from trust_assessor.scoring.engine import NOT_ASSESSED
from trust_assessor.services import analytics
from trust_assessor.services.assessment_service import assess_all, load_scoring_config
from trust_assessor.ui.components import bullet_list, card, heading
from trust_assessor.ui.layout import frame

LIMITATIONS = [
    "The Trust Score is a transparent heuristic. It is not validated against ground-truth labels of product trustworthiness, because none exist in the dataset.",
    "Suspicious-review signals flag reviews that deserve a closer look. Amazon Reviews 2023 has no fake-review labels, so no review is ever declared fake.",
    "Only a sample of one category is analysed (the first N lines of the file, optionally sub-sampled), so it may not represent the whole category or all of a product's reviews.",
    "Duplicate and user-behaviour signals only see reviews inside the sample; a reviewer's other reviews elsewhere are invisible.",
    "The default DistilBERT model is binary; the neutral class is derived from low confidence and sentiment on short or sarcastic reviews can be wrong.",
    "Review helpfulness is influenced by review age and visibility, so the informativeness component carries a deliberately low weight.",
    "Metadata is a snapshot. Products missing from the sampled metadata are scored without listing completeness rather than penalised.",
]


def register() -> None:
    @ui.page("/methodology")
    def methodology_page() -> None:
        with frame("Methodology & scoring settings", "methodology"):
            cfg = load_scoring_config()
            with card():
                heading("How the Product Trust Score works")
                ui.markdown(
                    "1. Review, listing and model outputs are turned into **eight components**, each normalised to 0-100.\n"
                    "2. Components with no data are **excluded** (never imputed) and the remaining weights are re-normalised. "
                    "The share of configured weight that had data is the **evidence coverage**.\n"
                    "3. The score is the **weighted average** of available components. A product with fewer than "
                    f"{cfg.min_reviews_for_score} reviews, or coverage below {cfg.min_coverage:.0%}, gets *no score*.\n"
                    f"4. A **red-flag cap** limits the score to {cfg.red_flag_score_cap:g} when authenticity signals score below "
                    f"{cfg.red_flag_authenticity_below:g} on products with at least {cfg.medium_conf_reviews} reviews "
                    "(strong ratings cannot fully offset heavy suspicious-review signals). It is always shown explicitly.\n"
                    f"5. Grades: **High** ≥ {cfg.grade_high:g}, **Moderate** ≥ {cfg.grade_moderate:g}, otherwise **Low**. "
                    f"Confidence is High with ≥ {cfg.high_conf_reviews} reviews and ≥ 90% coverage, Medium with ≥ {cfg.medium_conf_reviews} and ≥ 70%.")
            with card():
                heading("Components")
                for k in COMPONENT_KEYS:
                    with ui.row().classes("w-full no-wrap gap-3 py-1"):
                        ui.label(f"{cfg.normalised_weights()[k]:.0%}").classes("w-12 font-bold text-indigo-600 text-right")
                        with ui.column().classes("gap-0"):
                            ui.label(COMPONENT_LABELS[k]).classes("text-sm font-semibold")
                            ui.label(COMPONENT_DESCRIPTIONS[k]).classes("tc-sub")

            # ---------------------------------------------------------------- editor
            with card():
                heading("Scoring settings", "Weights are normalised to sum to 100%. Saving re-scores every product and appends a new run to the history.")
                sliders: dict[str, ui.slider] = {}
                labels: dict[str, ui.label] = {}
                with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 w-full mt-2"):
                    for k in COMPONENT_KEYS:
                        with ui.column().classes("gap-0 w-full"):
                            with ui.row().classes("justify-between w-full"):
                                ui.label(COMPONENT_LABELS[k]).classes("text-sm")
                                labels[k] = ui.label("").classes("text-sm font-mono")
                            sliders[k] = ui.slider(min=0, max=1, step=0.01, value=cfg.weights.get(k, DEFAULT_WEIGHTS[k]))

                def refresh_labels() -> None:
                    total = sum(s.value for s in sliders.values()) or 1.0
                    for k, s in sliders.items():
                        labels[k].set_text(f"raw {s.value:.2f} → {s.value / total:.1%}")

                for s in sliders.values():
                    s.on_value_change(refresh_labels)
                refresh_labels()

                with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-3 w-full mt-3"):
                    prior = ui.number("Bayesian prior weight", value=cfg.bayes_prior_weight, min=0, step=1).props("outlined dense")
                    target = ui.number("Volume target (ratings)", value=cfg.volume_target, min=10, step=10).props("outlined dense")
                    mincov = ui.number("Minimum coverage", value=cfg.min_coverage, min=0, max=1, step=0.05).props("outlined dense")
                    cap_on = ui.switch("Red-flag cap", value=cfg.red_flag_cap_enabled)
                status = ui.label("").classes("tc-sub")

                def current() -> ScoringConfig:
                    c = ScoringConfig.from_dict(cfg.to_dict())
                    c.weights = {k: float(s.value) for k, s in sliders.items()}
                    c.bayes_prior_weight, c.volume_target = float(prior.value or 0), float(target.value or 200)
                    c.min_coverage, c.red_flag_cap_enabled = float(mincov.value or 0), bool(cap_on.value)
                    return c

                async def save_and_rescore() -> None:
                    new = current()
                    if sum(new.weights.values()) <= 0:
                        ui.notify("At least one weight must be above zero.", type="negative")
                        return
                    new.save(get_settings().scoring_config_path)
                    status.set_text("Re-scoring all products …")
                    btn.disable()
                    try:
                        res = await run.io_bound(assess_all, None, new, "reweight")
                    finally:
                        btn.enable()
                    analytics.clear_cache()
                    status.set_text(f"Done: {res['n_products']:,} products re-scored (mean {res['mean_score']:.1f}), config {res['config_hash']}.")
                    ui.notify("Scoring configuration saved and applied.", type="positive")

                def reset() -> None:
                    for k, s in sliders.items():
                        s.set_value(DEFAULT_WEIGHTS[k])
                    d = ScoringConfig()
                    prior.set_value(d.bayes_prior_weight), target.set_value(d.volume_target)
                    mincov.set_value(d.min_coverage), cap_on.set_value(d.red_flag_cap_enabled)

                with ui.row().classes("gap-2 mt-2"):
                    btn = ui.button("Save & re-score all products", icon="calculate", on_click=save_and_rescore).props("color=primary")
                    ui.button("Reset to defaults", icon="restart_alt", on_click=reset).props("outline")

            with ui.element("div").classes("grid grid-cols-1 lg:grid-cols-2 gap-4 w-full"):
                with card():
                    heading("Limitations")
                    bullet_list(LIMITATIONS, "report_problem", "#d97706")
                with card():
                    heading("Deliberately not assessed", "No suitable data, so no invented numbers.")
                    bullet_list([f"{a}: {b}" for a, b in NOT_ASSESSED], "block", "#94a3b8")
                    ui.separator().classes("my-2")
                    heading("Data source")
                    ui.label("Amazon Reviews 2023 (McAuley Lab), streamed from Hugging Face. Hou et al., 2024, "
                             "'Bridging Language and Items for Retrieval and Recommendation' (arXiv:2403.03952). "
                             "Review text is analysed locally; nothing is scraped from live marketplaces.").classes("tc-sub")
