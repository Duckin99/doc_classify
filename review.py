"""
Document Classification Review Dashboard (v2 -- three-stage architecture)
==========================================================================
Run locally with:
    pip install streamlit pandas plotly
    streamlit run review_app_v2.py

Expects a results CSV (matching cascade_pipeline_v11.py's evaluate_experiment output)
with at least:
    filepath (or filename), ground_truth,
    triage_decision, triage_reason, triage_confidence,
    router_decision, router_reason, router_confidence,
    final_subcategory, specialist_reason, specialist_confidence

Optional ground-truth columns, auto-handled if missing/stale:
    triage_gt -- if missing, derived from ground_truth (medical_* -> "medical", else "non_medical").
        If present but still using the OLD 3-class scheme (medical / processable_non_medical /
        trash_others), it's automatically normalized to the new binary scheme.
    router_gt -- if missing, derived from ground_truth (financial_* -> "financial", id_* ->
        "identification", "eform" -> "eform", "unrelated_document" -> "unrelated_document",
        medical_* -> not applicable).

If ocr_text is missing, a clearly-labeled placeholder is generated automatically.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Doc Classification Review v2", layout="wide")

REQUIRED_BASE_COLS = [
    "triage_decision", "triage_reason", "triage_confidence",
    "router_decision", "router_reason", "router_confidence",
    "final_subcategory", "specialist_reason", "specialist_confidence",
    "ground_truth",
]

OLD_TRIAGE_GT_MAP = {
    "medical": "medical",
    "processable_non_medical": "non_medical",
    "trash_others": "non_medical",
}

ROUTER_NA_LABEL = "N/A (triage predicted medical)"


def infer_router_gt(leaf):
    """Derives the router-level ground truth from a leaf-level ground_truth label."""
    if not isinstance(leaf, str):
        return None
    if leaf.startswith("medical_"):
        return None  # not applicable -- medical docs never go through the router
    if leaf.startswith("financial_"):
        return "financial"
    if leaf.startswith("id_"):
        return "identification"
    if leaf == "eform":
        return "eform"
    if leaf == "unrelated_document":
        return "unrelated_document"
    return None  # unrecognized leaf (e.g. an error state) -- leave ungraded


def infer_triage_gt(leaf):
    """Derives the triage-level ground truth from a leaf-level ground_truth label."""
    if not isinstance(leaf, str):
        return None
    return "medical" if leaf.startswith("medical_") else "non_medical"


def prepare_ground_truth_columns(df: pd.DataFrame):
    """Adds/normalizes triage_gt and router_gt columns. Returns (df, list of notes)."""
    notes = []
    df = df.copy()

    if "triage_gt" not in df.columns:
        if "ground_truth" in df.columns:
            df["triage_gt"] = df["ground_truth"].apply(infer_triage_gt)
            notes.append("`triage_gt` derived from `ground_truth` (leaf label).")
        else:
            df["triage_gt"] = None
    else:
        unique_vals = set(df["triage_gt"].dropna().unique())
        if unique_vals - {"medical", "non_medical"}:
            df["triage_gt"] = df["triage_gt"].map(lambda v: OLD_TRIAGE_GT_MAP.get(v, v))
            notes.append("`triage_gt` normalized from the old 3-class scheme to the new binary scheme.")

    if "router_gt" not in df.columns:
        if "ground_truth" in df.columns:
            df["router_gt"] = df["ground_truth"].apply(infer_router_gt)
            notes.append("`router_gt` derived from `ground_truth` (leaf label).")
        else:
            df["router_gt"] = None

    return df, notes


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    has_ocr = "ocr_text" in df.columns
    if not has_ocr:
        path_col = "filepath" if "filepath" in df.columns else "filename"
        df["ocr_text"] = df[path_col].astype(str).apply(
            lambda fp: f"[OCR TEXT PLACEHOLDER -- column not wired up yet. File: {fp}]"
        )

    df, gt_notes = prepare_ground_truth_columns(df)
    return df, has_ocr, gt_notes


def build_confusion(df: pd.DataFrame, true_col: str, pred_col: str):
    labels = sorted(
        set(df[true_col].dropna().astype(str).unique())
        | set(df[pred_col].dropna().astype(str).unique())
    )
    cm = pd.crosstab(df[true_col].astype(str), df[pred_col].astype(str))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return cm, labels


def render_sample(row, true_col, pred_col, reason_cols, conf_cols, key_prefix, idx):
    header = f"{Path(str(row.get('filepath', row.get('filename', '')))).name}  —  GT: {row[true_col]}  →  Pred: {row[pred_col]}"
    with st.expander(header):
        img_col, text_col = st.columns([1, 1])

        with img_col:
            st.markdown("**Document image**")
            fp = row.get("filepath", row.get("filename", ""))
            try:
                st.image(fp, use_container_width=True)
            except Exception:
                st.info(f"Could not load image from path:\n`{fp}`")

        with text_col:
            st.markdown("**OCR text**")
            st.text_area(
                "ocr_text", value=str(row.get("ocr_text", "")),
                height=220, label_visibility="collapsed",
                key=f"{key_prefix}_ocr_{idx}",
            )

        for rc in reason_cols:
            if rc in row and pd.notna(row[rc]):
                st.markdown(f"**{rc}**")
                st.write(row[rc])

        conf_bits = [f"{c}: {row[c]}" for c in conf_cols if c in row and pd.notna(row[c])]
        if conf_bits:
            st.caption(" | ".join(conf_bits))


def render_matrix_tab(df: pd.DataFrame, true_col: str, pred_col: str,
                       reason_cols: list, conf_cols: list, title: str, key_prefix: str,
                       scope_note: str = ""):
    st.subheader(title)
    if scope_note:
        st.caption(scope_note)

    valid = df[[true_col, pred_col]].dropna(subset=[true_col])
    if len(valid) == 0:
        st.info("No rows with ground truth available for this stage.")
        return

    acc = (valid[true_col].astype(str) == valid[pred_col].astype(str)).mean()
    n_correct = int((valid[true_col].astype(str) == valid[pred_col].astype(str)).sum())
    st.metric("Accuracy", f"{acc:.2%}", help=f"{n_correct} / {len(valid)} correct")

    cm, labels = build_confusion(valid, true_col, pred_col)

    fig = px.imshow(
        cm.values,
        x=cm.columns, y=cm.index,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Ground Truth", color="Count"),
        aspect="auto",
    )
    fig.update_layout(height=max(420, 42 * len(labels)), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_heatmap")

    st.markdown("#### Drill into samples")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        row_choice = st.selectbox("Filter by Ground Truth (row)", ["All"] + labels, key=f"{key_prefix}_row")
    with c2:
        col_choice = st.selectbox("Filter by Prediction (column)", ["All"] + labels, key=f"{key_prefix}_col")
    with c3:
        only_errors = st.checkbox("Only misclassifications", value=True, key=f"{key_prefix}_err")

    filtered = valid.join(df.drop(columns=[true_col, pred_col]), how="left")
    if row_choice != "All":
        filtered = filtered[filtered[true_col].astype(str) == row_choice]
    if col_choice != "All":
        filtered = filtered[filtered[pred_col].astype(str) == col_choice]
    if only_errors:
        filtered = filtered[filtered[true_col].astype(str) != filtered[pred_col].astype(str)]

    st.write(f"**{len(filtered)}** matching sample(s)")

    max_show = 200
    for i, (idx, row) in enumerate(filtered.iterrows()):
        if i >= max_show:
            st.info(f"Showing first {max_show} of {len(filtered)} -- narrow your filter to see more precisely.")
            break
        render_sample(row, true_col, pred_col, reason_cols, conf_cols, key_prefix, idx)


def main():
    st.title("📄 Document Classification Review Dashboard (v2)")
    st.caption("Triage → Router → Specialist confusion matrices with per-sample drill-down (OCR text, image, reasoning, confidence).")

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload results CSV", type=["csv"])
    default_path = st.sidebar.text_input("...or a path on disk", value="")

    source = uploaded if uploaded is not None else (default_path or None)
    if source is None:
        st.info("Upload a CSV or provide a path in the sidebar to get started.")
        st.stop()

    df, has_ocr, gt_notes = load_data(source)

    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required column(s): {missing}")
        st.stop()

    if not has_ocr:
        st.sidebar.warning("No `ocr_text` column found -- showing placeholder text. Add the real column later, no code changes needed.")
    for note in gt_notes:
        st.sidebar.info(note)

    st.sidebar.metric("Total rows", len(df))

    # For the router-level matrix, only rows whose ground truth is actually non-medical
    # are in scope (medical docs never go through the router). Rows where triage
    # predicted "medical" but the true class was non-medical still belong in this matrix
    # -- their router_decision is NaN because the router was never called -- so we
    # substitute a visible sentinel instead of silently dropping them. Dropping would
    # hide exactly the cascading-failure cases you most want to see.
    router_scope = df[df["router_gt"].notna()].copy()
    router_scope["router_decision_display"] = router_scope["router_decision"].fillna(ROUTER_NA_LABEL)

    tab1, tab2, tab3 = st.tabs([
        "🩺 Step 1: Triage (medical / non_medical)",
        "🗂️ Step 2: Router (financial / identification / eform / unrelated_document)",
        "🔍 Step 3: Final Leaf Subclass",
    ])

    with tab1:
        render_matrix_tab(
            df, true_col="triage_gt", pred_col="triage_decision",
            reason_cols=["triage_reason"], conf_cols=["triage_confidence"],
            title="Triage: triage_gt vs triage_decision",
            key_prefix="triage",
        )

    with tab2:
        render_matrix_tab(
            router_scope, true_col="router_gt", pred_col="router_decision_display",
            reason_cols=["triage_reason", "router_reason"],
            conf_cols=["triage_confidence", "router_confidence"],
            title="Router: router_gt vs router_decision",
            key_prefix="router",
            scope_note=(
                f"Scoped to the {len(router_scope)} row(s) whose ground truth is non-medical "
                f"(medical docs never reach the router). Rows labeled '{ROUTER_NA_LABEL}' are cascading "
                f"failures -- triage incorrectly predicted 'medical', so the router was never called."
            ),
        )

    with tab3:
        render_matrix_tab(
            df, true_col="ground_truth", pred_col="final_subcategory",
            reason_cols=["triage_reason", "router_reason", "specialist_reason"],
            conf_cols=["triage_confidence", "router_confidence", "specialist_confidence"],
            title="Final Leaf: ground_truth vs final_subcategory",
            key_prefix="leaf",
            scope_note="End-to-end accuracy across all three stages combined.",
        )


if __name__ == "__main__":
    main()