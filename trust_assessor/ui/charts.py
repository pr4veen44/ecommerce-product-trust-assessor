"""Plotly figure builders. All functions are pure (data in, Figure out)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trust_assessor.ui.theme import GRADE_COLORS, GRID, INK, MUTED, PRIMARY, SERIES_COLORS, score_color

_FONT = dict(family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK, size=12)


def _base(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10), font=_FONT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def empty_fig(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height)


# ---------------------------------------------------------------- trust score
def gauge(score: float | None, cfg: dict[str, Any], height: int = 260) -> go.Figure:
    hi, mod = cfg.get("grade_high", 75), cfg.get("grade_moderate", 55)
    if score is None:
        return empty_fig("No score\n(insufficient evidence)", height)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=46, color=INK), valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=GRID),
            bar=dict(color=score_color(score, hi, mod), thickness=0.28),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, mod], color="#fee2e2"), dict(range=[mod, hi], color="#fef3c7"),
                   dict(range=[hi, 100], color="#dcfce7")],
        ),
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=0))
    return _base(fig, height)


def component_bars(components: Sequence[dict[str, Any]], height: int = 320) -> go.Figure:
    comps = list(components)[::-1]
    avail = [c for c in comps if c["available"]]
    fig = go.Figure()
    fig.add_bar(
        y=[c["label"] for c in avail], x=[c["score"] for c in avail], orientation="h",
        marker_color=[score_color(c["score"]) for c in avail],
        customdata=[[c["effective_weight"] * 100, c["contribution"]] for c in avail],
        hovertemplate="<b>%{y}</b><br>Score %{x:.0f}/100<br>Weight %{customdata[0]:.1f}%<br>Contribution %{customdata[1]:.1f} pts<extra></extra>",
        text=[f"{c['score']:.0f}" for c in avail], textposition="outside", cliponaxis=False,
    )
    unavailable = [c["label"] for c in comps if not c["available"]]
    fig.update_xaxes(range=[0, 110], gridcolor=GRID, title="Component score (0-100)")
    fig.update_yaxes(automargin=True)
    if unavailable:
        fig.add_annotation(text="No data: " + ", ".join(unavailable), xref="paper", yref="paper", x=0, y=-0.22,
                           showarrow=False, font=dict(color=MUTED, size=11), xanchor="left")
    return _base(fig, height)


def waterfall(assessment: dict[str, Any], height: int = 360) -> go.Figure:
    comps = [c for c in assessment["components"] if c["available"] and c["effective_weight"] > 0]
    if assessment["trust_score"] is None or not comps:
        return empty_fig("No score to decompose")
    labels = [c["label"] for c in comps]
    vals = [c["contribution"] for c in comps]
    measures = ["relative"] * len(comps)
    raw = assessment.get("raw_score")
    if assessment.get("adjustments") and raw is not None:
        labels.append("Red-flag cap")
        vals.append(assessment["trust_score"] - raw)
        measures.append("relative")
    labels.append("Trust Score")
    vals.append(assessment["trust_score"])
    measures.append("total")
    fig = go.Figure(go.Waterfall(
        x=labels, y=vals, measure=measures, text=[f"{v:+.1f}" if m == "relative" else f"{v:.1f}" for v, m in zip(vals, measures)],
        textposition="outside", connector=dict(line=dict(color=GRID)),
        increasing=dict(marker=dict(color="#6366f1")), decreasing=dict(marker=dict(color="#dc2626")),
        totals=dict(marker=dict(color=score_color(assessment["trust_score"]))),
    ))
    fig.update_yaxes(title="Points", gridcolor=GRID, range=[0, max(100, assessment["trust_score"] + 15)])
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height)


def radar(series: dict[str, Sequence[dict[str, Any]]], height: int = 380) -> go.Figure:
    fig = go.Figure()
    for i, (name, comps) in enumerate(series.items()):
        labels = [c["label"] for c in comps]
        vals = [c["score"] if c["available"] else 0 for c in comps]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], name=name, fill="toself", opacity=0.55,
            line=dict(color=color), hovertemplate="%{theta}: %{r:.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor=GRID), angularaxis=dict(tickfont=dict(size=10))))
    return _base(fig, height, legend=len(series) > 1)


def grouped_component_bars(series: dict[str, Sequence[dict[str, Any]]], height: int = 380) -> go.Figure:
    fig = go.Figure()
    for i, (name, comps) in enumerate(series.items()):
        fig.add_bar(name=name, x=[c["label"] for c in comps], y=[c["score"] if c["available"] else None for c in comps],
                    marker_color=SERIES_COLORS[i % len(SERIES_COLORS)])
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 100], gridcolor=GRID, title="Score")
    fig.update_xaxes(tickangle=-25)
    return _base(fig, height, legend=True)


# ---------------------------------------------------------------- reviews
def rating_distribution(product: pd.Series | dict[str, Any], height: int = 260) -> go.Figure:
    counts = [int(product.get(f"n_star_{i}", 0) or 0) for i in range(1, 6)]
    total = max(sum(counts), 1)
    fig = go.Figure(go.Bar(
        x=[f"{i}★" for i in range(1, 6)], y=counts, marker_color=["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"],
        text=[f"{c / total:.0%}" for c in counts], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} reviews<extra></extra>",
    ))
    fig.update_yaxes(gridcolor=GRID, title="Reviews")
    return _base(fig, height)


def sentiment_donut(product: pd.Series | dict[str, Any], height: int = 260) -> go.Figure:
    vals = [product.get("sent_pos_share"), product.get("sent_neu_share"), product.get("sent_neg_share")]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals[::2]):
        return empty_fig("Sentiment not available")
    neu = 0.0 if vals[1] is None or (isinstance(vals[1], float) and np.isnan(vals[1])) else vals[1]
    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"], values=[vals[0], neu, vals[2]], hole=0.6, sort=False,
        marker=dict(colors=["#22c55e", "#94a3b8", "#ef4444"]), textinfo="percent",
    ))
    return _base(fig, height, legend=True)


def rating_timeline(reviews: pd.DataFrame, height: int = 300) -> go.Figure:
    if len(reviews) < 3:
        return empty_fig("Not enough reviews for a timeline")
    ts = reviews.set_index("timestamp").sort_index()
    span_days = (ts.index.max() - ts.index.min()).days
    rule = "MS" if span_days <= 3 * 365 else "QS" if span_days <= 8 * 365 else "YS"
    agg = ts["rating"].resample(rule).agg(["mean", "count"]).dropna()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg.index, y=agg["count"], name="Reviews", marker_color="#c7d2fe", secondary_y=False)
    fig.add_scatter(x=agg.index, y=agg["mean"], name="Mean rating", mode="lines+markers",
                    line=dict(color=PRIMARY, width=2), secondary_y=True)
    fig.update_yaxes(title_text="Reviews", gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Mean ★", range=[1, 5.2], showgrid=False, secondary_y=True)
    return _base(fig, height, legend=True)


def rule_prevalence(product_rates: dict[str, float], dataset_rates: dict[str, float], height: int = 320) -> go.Figure:
    labels = list(product_rates)
    fig = go.Figure()
    fig.add_bar(y=labels, x=[product_rates[k] * 100 for k in labels], name="This product", orientation="h", marker_color=PRIMARY)
    fig.add_bar(y=labels, x=[dataset_rates.get(k, 0) * 100 for k in labels], name="Whole sample", orientation="h", marker_color="#cbd5e1")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="% of reviews", gridcolor=GRID)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _base(fig, height, legend=True)


# ---------------------------------------------------------------- dashboard
def score_histogram(scores: Sequence[float], cfg: dict[str, Any] | None = None, height: int = 300) -> go.Figure:
    if len(scores) == 0:
        return empty_fig("No scores yet")
    cfg = cfg or {}
    fig = go.Figure(go.Histogram(x=list(scores), xbins=dict(start=0, end=100, size=5), marker_color=PRIMARY, opacity=0.85))
    for x, name in ((cfg.get("grade_moderate", 55), "Moderate"), (cfg.get("grade_high", 75), "High")):
        fig.add_vline(x=x, line_dash="dot", line_color=MUTED, annotation_text=name, annotation_position="top")
    fig.update_xaxes(title="Trust Score", range=[0, 100], gridcolor=GRID)
    fig.update_yaxes(title="Products", gridcolor=GRID)
    return _base(fig, height)


def grade_donut(counts: dict[str, int], height: int = 300) -> go.Figure:
    order = [g for g in GRADE_COLORS if counts.get(g)]
    if not order:
        return empty_fig("No assessments yet")
    fig = go.Figure(go.Pie(labels=order, values=[counts[g] for g in order], hole=0.6, sort=False,
                           marker=dict(colors=[GRADE_COLORS[g] for g in order]), textinfo="percent"))
    return _base(fig, height, legend=True)


def mean_component_profile(means: dict[str, float], height: int = 300) -> go.Figure:
    labels = list(means)[::-1]
    fig = go.Figure(go.Bar(y=labels, x=[means[k] for k in labels], orientation="h",
                           marker_color=[score_color(means[k]) for k in labels],
                           text=[f"{means[k]:.0f}" for k in labels], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 110], gridcolor=GRID, title="Mean component score")
    fig.update_yaxes(automargin=True)
    return _base(fig, height)


def score_history(rows: list[dict[str, Any]], height: int = 280) -> go.Figure:
    pts = [(r["created_at"], r["trust_score"]) for r in rows if r["trust_score"] is not None][::-1]
    if not pts:
        return empty_fig("No scored assessments yet")
    fig = go.Figure(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                               line=dict(color=PRIMARY, width=2), marker=dict(size=8)))
    fig.update_yaxes(range=[0, 100], title="Trust Score", gridcolor=GRID)
    return _base(fig, height)


def run_trend(runs: list[dict[str, Any]], height: int = 280) -> go.Figure:
    pts = [(r["started"], r["mean_score"], r["source"]) for r in runs if r["mean_score"] is not None][::-1]
    if not pts:
        return empty_fig("No runs yet")
    fig = go.Figure(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", text=[p[2] for p in pts],
                               line=dict(color=PRIMARY), hovertemplate="%{x}<br>Mean score %{y:.1f} (%{text})<extra></extra>"))
    fig.update_yaxes(title="Mean Trust Score", gridcolor=GRID)
    return _base(fig, height)


# ---------------------------------------------------------------- model insights
def importance_bars(labels: Sequence[str], values: Sequence[float], title: str, color: str = PRIMARY, height: int = 380,
                    top: int = 15) -> go.Figure:
    order = np.argsort(values)[::-1][:top][::-1]
    fig = go.Figure(go.Bar(y=[labels[i] for i in order], x=[values[i] for i in order], orientation="h", marker_color=color))
    fig.update_xaxes(title=title, gridcolor=GRID)
    fig.update_yaxes(automargin=True)
    return _base(fig, height)


def shap_local_bars(items: Sequence[dict[str, Any]], label_key: str = "label", height: int = 320, flip: bool = False) -> go.Figure:
    items = list(items)[::-1]
    vals = [(-i["shap"] if flip else i["shap"]) for i in items]
    fig = go.Figure(go.Bar(
        y=[f"{i.get(label_key, i['feature'])} = {i['value']:.2f}" if i.get("value") is not None else str(i.get(label_key, i["feature"])) for i in items],
        x=vals, orientation="h", marker_color=["#16a34a" if v > 0 else "#dc2626" for v in vals],
    ))
    fig.update_xaxes(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED)
    fig.update_yaxes(automargin=True)
    return _base(fig, height)


def suspicion_hist(values: Sequence[float], threshold: float, height: int = 260) -> go.Figure:
    fig = go.Figure(go.Histogram(x=list(values), xbins=dict(start=0, end=1, size=0.05), marker_color="#a5b4fc"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#dc2626", annotation_text="flag threshold")
    fig.update_xaxes(title="Suspicion score", range=[0, 1], gridcolor=GRID)
    fig.update_yaxes(title="Reviews", type="log", gridcolor=GRID)
    return _base(fig, height)
