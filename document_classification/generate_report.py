"""
generate_report.py
===================
Builds a single self-contained static HTML report from pipeline output CSVs --
meant for sending to people who can't run Streamlit or pull the repo (`review.py`
and `flag_review.py` need `streamlit run`; this just needs a browser, no server).

The Plotly JS bundle is inlined once into the page, so the output HTML has zero
external/CDN dependencies -- it renders fully offline.

Requires: pandas, plotly (same stack as review.py / flag_review.py, minus streamlit).
    pip install pandas plotly

Usage
-----
    python3 generate_report.py --results results.csv --output report.html

    # include the multi-label imaging tagging stage
    python3 generate_report.py --results results.csv --flags flags.csv --output report.html

    # side-by-side comparison of two runs (e.g. multimodal vs text-only)
    python3 generate_report.py \\
        --results results_multimodal.csv --run-label Multimodal \\
        --compare-results results_text_only.csv --compare-label "Text-only" \\
        --flags flags_multimodal.csv --compare-flags flags_text_only.csv \\
        --imaging-routing imaging_routing_recall_multimodal.csv \\
        --compare-imaging-routing imaging_routing_recall_text_only.csv \\
        --output report.html

Expected columns
-----------------
--results (output of `document_classifier.py`):
    macro_gt, macro_decision, ground_truth, final_subcategory
    (optional) macro_latency_sec, macro_tokens_in, macro_tokens_out,
               specialist_latency_sec, specialist_tokens_in, specialist_tokens_out

--flags (output of `medical_imaging_flags.py`'s run_imaging_flags_batch):
    xray_gt, ultrasound_gt, ecg_gt,
    contains_xray, contains_ultrasound, contains_ecg
    (optional) latency_sec, tokens_in, tokens_out

--imaging-routing (output of `medical_imaging_flags.py`'s
evaluate_imaging_routing_recall(), only meaningful in --compare-results mode):
    final_subcategory, and at least one of xray_gt/ultrasound_gt/ecg_gt
    Renders one bar chart in the Head-to-Head Comparison section: of documents that
    truly have an imaging finding, what fraction the cascade routed somewhere
    medical_imaging_flags.py would actually run on it. Different question from
    --flags (which measures the imaging tagger's own precision/recall).

Missing optional columns degrade gracefully (that chart/row is just skipped).
"""

import argparse
import html as htmlmod
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

MULTILABEL_TAGS = ("xray", "ultrasound", "ecg")
DOMAIN_ORDER = ["medical", "financial", "identification", "not_for_underwriting"]
SPECIALIST_DOMAINS = ["medical", "financial", "identification"]  # domains with a specialist stage

BLUE, PURPLE, GREEN = "#2563eb", "#7c3aed", "#059669"
# ColorBrewer "Blues" tints, same family as the confusion-matrix heatmap colorscale.
# Every bar chart in the report draws from this palette so it reads as one system.
BLUES_DARK, BLUES_MID, BLUES_LIGHT, BLUES_PALE = "#08306b", "#2171b5", "#4292c6", "#9ecae1"
CMP_A, CMP_B = BLUES_DARK, BLUES_LIGHT  # two-run comparison charts: primary run vs. comparison run


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(s):
    return htmlmod.escape("" if s is None else str(s))


def ordered_labels(labels):
    known = [d for d in DOMAIN_ORDER if d in labels]
    rest = sorted(l for l in labels if l not in DOMAIN_ORDER)
    return known + rest


def fig_to_div(fig, div_id):
    # pyo.plot() mints its own random UUID div id per call, already unique across figures.
    return pyo.plot(
        fig, output_type="div", include_plotlyjs=False, config={"displaylogo": False, "responsive": True},
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_confusion(df, true_col, pred_col):
    sub = df[[true_col, pred_col]].dropna()
    sub = sub[(sub[true_col] != "") & (sub[pred_col] != "")]
    if sub.empty:
        return None, [], 0
    labels = ordered_labels(set(sub[true_col].astype(str)) | set(sub[pred_col].astype(str)))
    cm = pd.crosstab(sub[true_col].astype(str), sub[pred_col].astype(str))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return cm, labels, len(sub)


def class_metrics_from_cm(cm, labels):
    rows = []
    for label in labels:
        tp = int(cm.loc[label, label])
        fp = int(cm[label].sum() - tp)
        fn = int(cm.loc[label].sum() - tp)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"label": label, "support": support, "precision": precision, "recall": recall, "f1": f1})
    return rows


def accuracy_from_cm(cm):
    total = cm.values.sum()
    correct = sum(cm.loc[l, l] for l in cm.index if l in cm.columns)
    return (correct / total) if total else 0.0


def macro_avg(metrics, key):
    return sum(m[key] for m in metrics) / len(metrics) if metrics else 0.0


def specialist_breakdown(df):
    """Per-domain specialist performance, isolated from macro-stage routing errors.

    Restricts to rows where macro_decision == macro_gt (the macro stage routed
    to the correct domain), then measures final_subcategory vs ground_truth
    within each domain. Misrouted documents are excluded rather than counted
    as specialist misses -- a document sent to the wrong specialist was never
    going to get the right subclass label regardless of that specialist's
    quality, so including it would blend stage-1 and stage-2 error together.
    """
    needed = {"macro_gt", "macro_decision", "ground_truth", "final_subcategory"}
    if not needed.issubset(df.columns):
        return {}
    sub = df[list(needed)].dropna()
    if sub.empty:
        return {}
    routed = sub[sub["macro_decision"] == sub["macro_gt"]]
    out = {}
    for domain, g in routed.groupby("macro_gt"):
        if domain not in SPECIALIST_DOMAINS:
            continue  # not_for_underwriting is terminal at macro -- no specialist, no subclass to score
        correct = int((g["ground_truth"] == g["final_subcategory"]).sum())
        cm, labels, n = build_confusion(g, "ground_truth", "final_subcategory")
        out[domain] = {
            "accuracy": correct / len(g) if len(g) else 0.0,
            "n": len(g),
            "cm": cm, "labels": labels,
            "class_metrics": class_metrics_from_cm(cm, labels) if cm is not None else [],
        }
    return out


def multilabel_metrics(df, tags=MULTILABEL_TAGS):
    if df is None:
        return None
    per_tag = {}
    for tag in tags:
        gt_col, pred_col = f"{tag}_gt", f"contains_{tag}"
        if gt_col not in df.columns or pred_col not in df.columns:
            continue
        sub = df[[gt_col, pred_col]].dropna()
        if sub.empty:
            continue
        gt = sub[gt_col].astype(bool)
        pred = sub[pred_col].astype(bool)
        tp = int((gt & pred).sum())
        fp = int((~gt & pred).sum())
        fn = int((gt & ~pred).sum())
        tn = int((~gt & ~pred).sum())
        n = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / n if n else 0.0
        per_tag[tag] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
                         "precision": precision, "recall": recall, "f1": f1,
                         "accuracy": accuracy, "support": tp + fn}
    if not per_tag:
        return None
    cols = [(f"{t}_gt", f"contains_{t}") for t in per_tag]
    sub = df.dropna(subset=[c for pair in cols for c in pair])
    exact = sum(
        all(bool(row[g]) == bool(row[p]) for g, p in cols)
        for _, row in sub.iterrows()
    ) if len(sub) else 0
    return {"per_tag": per_tag, "exact_match_ratio": exact / len(sub) if len(sub) else 0.0, "n": len(sub)}


IMAGING_ELIGIBLE_SUBCATEGORIES = {"medical_clinical", "medical_healthcheck", "medical_lab"}


def imaging_routing_recall(df):
    """Of documents that truly have at least one imaging finding (xray_gt /
    ultrasound_gt / ecg_gt), what fraction did the macro+specialist cascade route
    into a subclass medical_imaging_flags.py actually runs against
    (IMAGING_ELIGIBLE_SUBCATEGORIES)? A misrouted imaging-positive document never
    reaches the imaging tagger at all, so this is upstream routing recall, not an
    imaging-tagging accuracy number -- expects the output of
    medical_imaging_flags.evaluate_imaging_routing_recall(), not the regular
    --results or --flags CSVs.
    """
    if df is None or "final_subcategory" not in df.columns:
        return None
    gt_cols = [c for c in ("xray_gt", "ultrasound_gt", "ecg_gt") if c in df.columns]
    if not gt_cols:
        return None
    sub = df.dropna(subset=["final_subcategory"])
    if sub.empty:
        return None
    has_flag = sub[gt_cols].fillna(False).astype(bool).any(axis=1)
    positive = sub[has_flag]
    if positive.empty:
        return None
    captured = positive["final_subcategory"].isin(IMAGING_ELIGIBLE_SUBCATEGORIES)
    return {"recall": float(captured.mean()), "n": len(positive), "captured": int(captured.sum())}


def latency_token_frame(df):
    out = pd.DataFrame(index=df.index)
    for col in ("macro_latency_sec", "specialist_latency_sec", "macro_tokens_in",
                "specialist_tokens_in", "macro_tokens_out", "specialist_tokens_out"):
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA
    out["e2e_latency_sec"] = out["macro_latency_sec"].fillna(0) + out["specialist_latency_sec"].fillna(0)
    out["e2e_tokens_in"] = out["macro_tokens_in"].fillna(0) + out["specialist_tokens_in"].fillna(0)
    out["e2e_tokens_out"] = out["macro_tokens_out"].fillna(0) + out["specialist_tokens_out"].fillna(0)
    return out


def pipeline_summary(df):
    macro_cm, macro_labels, macro_n = build_confusion(df, "macro_gt", "macro_decision")
    e2e_cm, e2e_labels, e2e_n = build_confusion(df, "ground_truth", "final_subcategory")
    macro_class_m = class_metrics_from_cm(macro_cm, macro_labels) if macro_cm is not None else []
    e2e_class_m = class_metrics_from_cm(e2e_cm, e2e_labels) if e2e_cm is not None else []
    lt = latency_token_frame(df)
    return {
        "n": len(df),
        "macro": {
            "cm": macro_cm, "labels": macro_labels, "n": macro_n,
            "accuracy": accuracy_from_cm(macro_cm) if macro_cm is not None else 0.0,
            "class_metrics": macro_class_m,
            "macro_precision": macro_avg(macro_class_m, "precision"),
            "macro_recall": macro_avg(macro_class_m, "recall"),
            "macro_f1": macro_avg(macro_class_m, "f1"),
        },
        "e2e": {
            "cm": e2e_cm, "labels": e2e_labels, "n": e2e_n,
            "accuracy": accuracy_from_cm(e2e_cm) if e2e_cm is not None else 0.0,
            "class_metrics": e2e_class_m,
            "macro_f1": macro_avg(e2e_class_m, "f1"),
            "by_domain": specialist_breakdown(df),
        },
        "lt": lt,
        "lt_means": {
            "macro_latency_sec": lt["macro_latency_sec"].mean(),
            "specialist_latency_sec": lt["specialist_latency_sec"].mean(),
            "e2e_latency_sec": lt["e2e_latency_sec"].mean(),
            "e2e_tokens_in": lt["e2e_tokens_in"].mean(),
            "e2e_tokens_out": lt["e2e_tokens_out"].mean(),
        },
    }


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def confusion_heatmap_fig(cm, labels, true_name, pred_name):
    fig = go.Figure(data=go.Heatmap(
        z=cm.values, x=labels, y=labels, colorscale="Blues", showscale=False,
        text=cm.values, texttemplate="%{text}", hovertemplate=f"{true_name}: %{{y}}<br>{pred_name}: %{{x}}<br>Count: %{{z}}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=pred_name, side="bottom"), yaxis=dict(title=true_name, autorange="reversed"),
        height=max(340, 46 * len(labels)), margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def metrics_bar_fig(metrics):
    dfm = pd.DataFrame(metrics)
    fig = go.Figure(data=[
        go.Bar(name="Precision", x=dfm["label"], y=dfm["precision"], marker_color=BLUES_DARK,
               text=dfm["precision"], texttemplate="%{text:.0%}", textposition="outside"),
        go.Bar(name="Recall", x=dfm["label"], y=dfm["recall"], marker_color=BLUES_MID,
               text=dfm["recall"], texttemplate="%{text:.0%}", textposition="outside"),
        go.Bar(name="F1", x=dfm["label"], y=dfm["f1"], marker_color=BLUES_LIGHT,
               text=dfm["f1"], texttemplate="%{text:.0%}", textposition="outside"),
    ])
    fig.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 1.18],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def domain_accuracy_fig(by_domain):
    domains = ordered_labels(by_domain.keys())
    accs = [by_domain[d]["accuracy"] for d in domains]
    ns = [by_domain[d]["n"] for d in domains]
    fig = go.Figure(go.Bar(
        x=domains, y=accs, marker_color=BLUES_MID, text=[f"{a:.1%} (n={n})" for a, n in zip(accs, ns)],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_tickformat=".0%", yaxis_range=[0, 1.15], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10), height=340,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def latency_box_fig(lt):
    fig = go.Figure()
    for name, col, color in [("Macro", "macro_latency_sec", PURPLE),
                              ("Specialist", "specialist_latency_sec", BLUE),
                              ("End-to-End", "e2e_latency_sec", GREEN)]:
        vals = lt[col].dropna()
        if len(vals):
            fig.add_trace(go.Box(y=vals, name=name, marker_color=color, boxmean=True))
    fig.update_layout(
        yaxis_title="Seconds", showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10), height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def token_box_fig(lt):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Input tokens (e2e)", "Output tokens (e2e)"))
    fig.add_trace(go.Box(y=lt["e2e_tokens_in"].dropna(), name="Input", marker_color=BLUE, boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=lt["e2e_tokens_out"].dropna(), name="Output", marker_color=GREEN, boxmean=True), row=1, col=2)
    fig.update_layout(
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def small_cm_fig(m):
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["Pred: False", "Pred: True"], y=["GT: False", "GT: True"],
        colorscale="Blues", showscale=False, text=z, texttemplate="%{text}",
    ))
    # Fixed pixel size (not responsive) -- these sit in a wrapping flex grid, and
    # letting Plotly auto-size to its container is what caused the overlap: the
    # container's width isn't settled yet at the moment this script tag runs, so
    # Plotly falls back to its ~700px default canvas and spills into neighbors.
    fig.update_layout(width=320, height=280, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis=dict(autorange="reversed"),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def multilabel_bar_fig(per_tag):
    tags = list(per_tag.keys())
    fig = go.Figure(data=[
        go.Bar(name="Accuracy", x=[t.upper() for t in tags], y=[per_tag[t]["accuracy"] for t in tags], marker_color=BLUES_DARK,
               text=[per_tag[t]["accuracy"] for t in tags], texttemplate="%{text:.0%}", textposition="outside"),
        go.Bar(name="Precision", x=[t.upper() for t in tags], y=[per_tag[t]["precision"] for t in tags], marker_color=BLUES_MID,
               text=[per_tag[t]["precision"] for t in tags], texttemplate="%{text:.0%}", textposition="outside"),
        go.Bar(name="Recall", x=[t.upper() for t in tags], y=[per_tag[t]["recall"] for t in tags], marker_color=BLUES_LIGHT,
               text=[per_tag[t]["recall"] for t in tags], texttemplate="%{text:.0%}", textposition="outside"),
        go.Bar(name="F1", x=[t.upper() for t in tags], y=[per_tag[t]["f1"] for t in tags], marker_color=BLUES_PALE,
               text=[per_tag[t]["f1"] for t in tags], texttemplate="%{text:.0%}", textposition="outside"),
    ])
    fig.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 1.18],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10), height=400,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def comparison_bar_fig(label_a, summary_a, label_b, summary_b):
    metrics = ["Macro accuracy", "End-to-end accuracy"]
    vals_a = [summary_a["macro"]["accuracy"], summary_a["e2e"]["accuracy"]]
    vals_b = [summary_b["macro"]["accuracy"], summary_b["e2e"]["accuracy"]]
    fig = go.Figure(data=[
        go.Bar(name=label_a, x=metrics, y=vals_a, marker_color=CMP_A, text=vals_a, texttemplate="%{text:.1%}", textposition="outside"),
        go.Bar(name=label_b, x=metrics, y=vals_b, marker_color=CMP_B, text=vals_b, texttemplate="%{text:.1%}", textposition="outside"),
    ])
    fig.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 1.18],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def comparison_specialist_fig(label_a, summary_a, label_b, summary_b):
    """One grouped bar chart, per-specialist accuracy (medical/financial/identification),
    run A vs run B -- restricted to correctly-routed documents, same definition as
    the single-run 'Specialist accuracy by domain' chart."""
    domains = [d for d in SPECIALIST_DOMAINS
               if d in summary_a["e2e"]["by_domain"] or d in summary_b["e2e"]["by_domain"]]
    vals_a = [summary_a["e2e"]["by_domain"].get(d, {}).get("accuracy", 0.0) for d in domains]
    vals_b = [summary_b["e2e"]["by_domain"].get(d, {}).get("accuracy", 0.0) for d in domains]
    fig = go.Figure(data=[
        go.Bar(name=label_a, x=[d.title() for d in domains], y=vals_a, marker_color=CMP_A,
               text=vals_a, texttemplate="%{text:.1%}", textposition="outside"),
        go.Bar(name=label_b, x=[d.title() for d in domains], y=vals_b, marker_color=CMP_B,
               text=vals_b, texttemplate="%{text:.1%}", textposition="outside"),
    ])
    fig.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 1.18],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def imaging_routing_recall_fig(label_a, recall_a, label_b, recall_b):
    fig = go.Figure(go.Bar(
        x=[label_a, label_b], y=[recall_a["recall"], recall_b["recall"]],
        marker_color=[CMP_A, CMP_B],
        text=[f"{recall_a['recall']:.1%} ({recall_a['captured']}/{recall_a['n']})",
              f"{recall_b['recall']:.1%} ({recall_b['captured']}/{recall_b['n']})"],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_tickformat=".0%", yaxis_range=[0, 1.18], showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10), height=360,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,130,145,0.18)")
    return fig


def comparison_latency_token_fig(label_a, summary_a, label_b, summary_b):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Avg e2e latency (s)", "Avg e2e tokens in", "Avg e2e tokens out"))
    m_a, m_b = summary_a["lt_means"], summary_b["lt_means"]
    pairs = [
        ("e2e_latency_sec", 1), ("e2e_tokens_in", 2), ("e2e_tokens_out", 3),
    ]
    for key, col in pairs:
        fig.add_trace(go.Bar(x=[label_a, label_b], y=[m_a[key], m_b[key]],
                              marker_color=[CMP_A, CMP_B], showlegend=False,
                              text=[f"{m_a[key]:.2f}", f"{m_b[key]:.2f}"], textposition="outside"),
                      row=1, col=col)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=10, r=10, t=40, b=10), height=340)
    return fig


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def stat_card(label, value, sub=None):
    sub_html = f"<div class='stat-sub'>{esc(sub)}</div>" if sub else ""
    return f"""
    <div class="stat-card">
      <div class="stat-value">{esc(value)}</div>
      <div class="stat-label">{esc(label)}</div>
      {sub_html}
    </div>"""


def metrics_table_html(metrics):
    rows_html = "".join(
        f"<tr><td>{esc(m['label'])}</td><td>{m['support']}</td>"
        f"<td>{m['precision']:.1%}</td><td>{m['recall']:.1%}</td><td>{m['f1']:.1%}</td></tr>"
        for m in metrics
    )
    return f"""
    <div class="table-scroll">
    <table class="metrics">
      <thead><tr><th>Label</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>"""


def specialist_confusion_section_html(by_domain, div_prefix):
    domains = [d for d in ordered_labels(by_domain.keys()) if d in SPECIALIST_DOMAINS]
    if not domains:
        return ""
    blocks = []
    for domain in domains:
        d = by_domain[domain]
        if d["cm"] is None:
            blocks.append(f"<h4>{esc(domain.title())} specialist</h4><p class='muted'>No correctly-routed rows for this domain.</p>")
            continue
        blocks.append(f"""
        <h4>{esc(domain.title())} specialist &mdash; {d['accuracy']:.1%} accuracy (n={d['n']})</h4>
        {fig_to_div(confusion_heatmap_fig(d['cm'], d['labels'], 'ground_truth', 'final_subcategory'), f"{div_prefix}_spec_{domain}")}
        {metrics_table_html(d['class_metrics'])}
        """)
    return f"""
    <h3>Per-specialist confusion matrices</h3>
    <p class="muted">Restricted to documents the macro stage routed to the correct domain, so each matrix reflects that specialist's own subclass accuracy in isolation from routing errors.</p>
    {''.join(blocks)}
    """


def multilabel_section_html(ml, div_prefix):
    if not ml:
        return ""
    cm_blocks = "".join(
        f'<div class="cm-small"><h4>{esc(tag.upper())}</h4>{fig_to_div(small_cm_fig(m), f"{div_prefix}_cm_{tag}")}</div>'
        for tag, m in ml["per_tag"].items()
    )
    return f"""
    <section>
      <h2>Stage 3 &mdash; Multi-Label Medical Diagnostics (X-Ray / Ultrasound / ECG)</h2>
      <div class="stat-row">
        {stat_card("Exact Match Ratio", f"{ml['exact_match_ratio']:.1%}", f"all labels correct, n={ml['n']}")}
      </div>
      {fig_to_div(multilabel_bar_fig(ml['per_tag']), f"{div_prefix}_ml_bar")}
      <h3>Per-modality confusion matrices</h3>
      <div class="cm-grid">{cm_blocks}</div>
    </section>
    """


def run_section_html(label, summary, ml, div_prefix):
    macro, e2e = summary["macro"], summary["e2e"]
    macro_cm_html = (
        fig_to_div(confusion_heatmap_fig(macro["cm"], macro["labels"], "macro_gt", "macro_decision"), f"{div_prefix}_macro_cm")
        if macro["cm"] is not None else "<p class='muted'>No overlapping rows for macro_gt / macro_decision.</p>"
    )
    e2e_cm_html = (
        fig_to_div(confusion_heatmap_fig(e2e["cm"], e2e["labels"], "ground_truth", "final_subcategory"), f"{div_prefix}_e2e_cm")
        if e2e["cm"] is not None else "<p class='muted'>No overlapping rows for ground_truth / final_subcategory.</p>"
    )
    domain_fig_html = (
        fig_to_div(domain_accuracy_fig(e2e["by_domain"]), f"{div_prefix}_domain")
        if e2e["by_domain"] else "<p class='muted'>No domain breakdown available.</p>"
    )
    lat_mean = summary["lt_means"]["e2e_latency_sec"]
    tok_in_mean = summary["lt_means"]["e2e_tokens_in"]
    tok_out_mean = summary["lt_means"]["e2e_tokens_out"]

    return f"""
    <section>
      <h2>{esc(label)}</h2>
      <div class="stat-row">
        {stat_card("Macro Accuracy", f"{macro['accuracy']:.1%}", f"n={macro['n']}")}
        {stat_card("End-to-End Accuracy", f"{e2e['accuracy']:.1%}", f"n={e2e['n']}")}
        {stat_card("Avg. E2E Latency", f"{lat_mean:.2f}s")}
        {stat_card("Avg. E2E Tokens", f"{tok_in_mean:,.0f} in / {tok_out_mean:,.0f} out")}
      </div>

      <h3>Stage 1 &mdash; Macro Triage (macro_gt vs macro_decision)</h3>
      {macro_cm_html}
      {metrics_table_html(macro['class_metrics']) if macro['class_metrics'] else ''}
      {fig_to_div(metrics_bar_fig(macro['class_metrics']), f"{div_prefix}_macro_bar") if macro['class_metrics'] else ''}

      <h3>Stage 2 &mdash; End-to-End Subcategory (ground_truth vs final_subcategory)</h3>
      {e2e_cm_html}
      {metrics_table_html(e2e['class_metrics']) if e2e['class_metrics'] else ''}
      {fig_to_div(metrics_bar_fig(e2e['class_metrics']), f"{div_prefix}_e2e_bar") if e2e['class_metrics'] else ''}

      <h3>Specialist accuracy by domain</h3>
      <p class="muted">Restricted to documents the macro stage routed to the correct domain -- isolates specialist performance from macro routing errors.</p>
      {domain_fig_html}

      {specialist_confusion_section_html(e2e['by_domain'], div_prefix)}

      <h3>Latency &amp; token distribution</h3>
      {fig_to_div(latency_box_fig(summary['lt']), f"{div_prefix}_lat_box")}
      {fig_to_div(token_box_fig(summary['lt']), f"{div_prefix}_tok_box")}

      {multilabel_section_html(ml, div_prefix)}
    </section>
    """


def comparison_section_html(label_a, summary_a, label_b, summary_b, recall_a=None, recall_b=None):
    imaging_recall_html = ""
    if recall_a and recall_b:
        imaging_recall_html = f"""
      <h3>Imaging routing recall</h3>
      <p class="muted">Of documents that truly contain an X-ray/ultrasound/ECG finding, the share the macro+specialist cascade routed into a subclass medical_imaging_flags.py actually runs against (medical_clinical/medical_healthcheck/medical_lab). A misrouted imaging-positive document never reaches the imaging tagger at all -- this is upstream routing recall, not imaging-tagging accuracy.</p>
      {fig_to_div(imaging_routing_recall_fig(label_a, recall_a, label_b, recall_b), "cmp_ir")}
        """
    return f"""
    <section>
      <h2>Head-to-Head Comparison</h2>
      {fig_to_div(comparison_bar_fig(label_a, summary_a, label_b, summary_b), "cmp_acc")}
      <h3>Specialist accuracy by domain</h3>
      <p class="muted">Restricted to documents the macro stage routed to the correct domain -- isolates specialist performance from macro routing errors.</p>
      {fig_to_div(comparison_specialist_fig(label_a, summary_a, label_b, summary_b), "cmp_spec")}
      {imaging_recall_html}
      <h3>Latency &amp; tokens</h3>
      {fig_to_div(comparison_latency_token_fig(label_a, summary_a, label_b, summary_b), "cmp_lt")}
    </section>
    """


CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  margin: 0; padding: 0 0 4rem 0;
  background: #f7f8fa; color: #1a1d23;
}
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e6e8eb; }
  .stat-card, table.metrics, .cm-small { background: #1c1f26 !important; border-color: #2b2f38 !important; }
  th { color: #9aa3af !important; }
  .muted { color: #7b8290 !important; }
}
header.report-header {
  padding: 2.5rem 1.5rem 2rem; background: linear-gradient(135deg,#2563eb,#1e3a8a); color: #fff;
}
header.report-header .wrap { max-width: 1100px; margin: 0 auto; }
header.report-header h1 { margin: 0 0 0.35rem; font-size: 1.6rem; }
header.report-header p { margin: 0.15rem 0; opacity: 0.9; font-size: 0.92rem; }
main { max-width: 1100px; margin: 0 auto; padding: 0 1.5rem; }
section { margin-top: 2.5rem; }
h2 { font-size: 1.25rem; border-bottom: 2px solid #2563eb; padding-bottom: 0.4rem; }
h3 { font-size: 1.02rem; margin-top: 1.8rem; color: #374151; }
@media (prefers-color-scheme: dark) { h3 { color: #b7bec9; } }
.stat-row { display: flex; flex-wrap: wrap; gap: 0.9rem; margin: 1rem 0 1.5rem; }
.stat-card {
  background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.9rem 1.1rem;
  min-width: 150px; flex: 1 1 150px;
}
.stat-value { font-size: 1.5rem; font-weight: 700; color: #2563eb; }
.stat-label { font-size: 0.8rem; color: #6b7280; margin-top: 0.15rem; }
.stat-sub { font-size: 0.72rem; color: #9ca3af; margin-top: 0.2rem; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; font-size: 0.86rem; }
table.metrics { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
th, td { padding: 0.45rem 0.65rem; text-align: center; border-bottom: 1px solid #eef0f3; white-space: nowrap; }
th { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; color: #6b7280; }
td:first-child, th:first-child { text-align: left; }
.cm-grid { display: flex; flex-wrap: wrap; gap: 1.2rem; }
.cm-small { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.8rem; flex: 0 0 auto; }
.cm-small h4 { margin: 0 0 0.5rem; font-size: 0.85rem; }
.muted { color: #6b7280; font-size: 0.85rem; }
footer { max-width: 1100px; margin: 3rem auto 0; padding: 0 1.5rem; color: #9ca3af; font-size: 0.78rem; }
"""


def build_html(title, generated_at, sections_html, n_runs_note, plotly_js):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
<script>{plotly_js}</script>
</head>
<body>
<header class="report-header">
  <div class="wrap">
    <h1>{esc(title)}</h1>
    <p>Generated {esc(generated_at)}</p>
    <p>{esc(n_runs_note)}</p>
  </div>
</header>
<main>
{sections_html}
</main>
<footer>Generated by generate_report.py &mdash; static report, Plotly bundled inline, no network requests.</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="Main results CSV (document_classifier.py output)")
    ap.add_argument("--flags", help="Multi-label imaging flags CSV (medical_imaging_flags.py output)")
    ap.add_argument("--imaging-routing", help="Output of medical_imaging_flags.evaluate_imaging_routing_recall() for the primary run")
    ap.add_argument("--run-label", default="Run", help="Label for the primary run (used in headings / comparison table)")
    ap.add_argument("--compare-results", help="Optional second results CSV to compare against --results")
    ap.add_argument("--compare-flags", help="Optional second flags CSV to pair with --compare-results")
    ap.add_argument("--compare-imaging-routing", help="Output of evaluate_imaging_routing_recall() for the comparison run")
    ap.add_argument("--compare-label", default="Comparison Run", help="Label for the comparison run")
    ap.add_argument("--title", default="Document Classification -- Evaluation Report", help="Report <title> / heading")
    ap.add_argument("--output", default="classification_report.html", help="Output HTML file path")
    args = ap.parse_args()

    df_a = pd.read_csv(args.results)
    flags_a = pd.read_csv(args.flags) if args.flags else None
    ir_a = pd.read_csv(args.imaging_routing) if args.imaging_routing else None
    summary_a = pipeline_summary(df_a)
    ml_a = multilabel_metrics(flags_a) if flags_a is not None else None
    recall_a = imaging_routing_recall(ir_a) if ir_a is not None else None

    sections = [run_section_html(args.run_label, summary_a, ml_a, "a")]
    note = f"{len(df_a)} documents ({args.run_label})"

    if args.compare_results:
        df_b = pd.read_csv(args.compare_results)
        flags_b = pd.read_csv(args.compare_flags) if args.compare_flags else None
        ir_b = pd.read_csv(args.compare_imaging_routing) if args.compare_imaging_routing else None
        summary_b = pipeline_summary(df_b)
        ml_b = multilabel_metrics(flags_b) if flags_b is not None else None
        recall_b = imaging_routing_recall(ir_b) if ir_b is not None else None
        sections.insert(0, comparison_section_html(args.run_label, summary_a, args.compare_label, summary_b, recall_a, recall_b))
        sections.append(run_section_html(args.compare_label, summary_b, ml_b, "b"))
        note += f" vs {len(df_b)} documents ({args.compare_label})"

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    plotly_js = pyo.get_plotlyjs()
    out_html = build_html(args.title, generated_at, "\n".join(sections), note, plotly_js)

    out_path = Path(args.output)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"Wrote {out_path.resolve()} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
