"""
Document Classification Review Dashboard
=========================================
Run locally with:
    pip install streamlit pandas plotly
    streamlit run review_app.py

Expects a CSV with (at minimum) these columns:
    filepath, macro_decision, macro_reason, macro_confidence,
    final_subcategory, specialist_reason, specialist_confidence,
    ground_truth, macro_gt

If an `ocr_text` column is missing, a clearly-labeled placeholder is
generated automatically so the UI works today -- once you wire up the real
OCR text column, it will just appear with no code changes needed.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Doc Classification Review", layout="wide")

REQUIRED_COLS = [
    "filepath", "macro_decision", "macro_reason", "macro_confidence",
    "final_subcategory", "specialist_reason", "specialist_confidence",
    "ground_truth", "macro_gt",
]


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    has_ocr = "ocr_text" in df.columns
    if not has_ocr:
        df["ocr_text"] = df["filepath"].astype(str).apply(
            lambda fp: f"[OCR TEXT PLACEHOLDER -- column not wired up yet. File: {fp}]"
        )
    return df, has_ocr


def build_confusion(df: pd.DataFrame, true_col: str, pred_col: str):
    labels = sorted(
        set(df[true_col].dropna().astype(str).unique())
        | set(df[pred_col].dropna().astype(str).unique())
    )
    cm = pd.crosstab(df[true_col].astype(str), df[pred_col].astype(str))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return cm, labels


def render_sample(row, true_col, pred_col, reason_cols, conf_cols, key_prefix, idx):
    header = f"{Path(str(row.get('filepath', ''))).name}  —  GT: {row[true_col]}  →  Pred: {row[pred_col]}"
    with st.expander(header):
        img_col, text_col = st.columns([1, 1])

        with img_col:
            st.markdown("**Document image**")
            fp = row.get("filepath", "")
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
            if rc in row:
                st.markdown(f"**{rc}**")
                st.write(row[rc])

        conf_bits = [f"{c}: {row[c]}" for c in conf_cols if c in row]
        if conf_bits:
            st.caption(" | ".join(conf_bits))


def render_matrix_tab(df: pd.DataFrame, true_col: str, pred_col: str,
                       reason_cols: list, conf_cols: list, title: str, key_prefix: str):
    st.subheader(title)

    valid = df[[true_col, pred_col]].dropna()
    acc = (valid[true_col].astype(str) == valid[pred_col].astype(str)).mean() if len(valid) else 0
    n_correct = int((valid[true_col].astype(str) == valid[pred_col].astype(str)).sum())
    st.metric("Accuracy", f"{acc:.2%}", help=f"{n_correct} / {len(valid)} correct")

    cm, labels = build_confusion(df, true_col, pred_col)

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

    filtered = df.copy()
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
    st.title("📄 Document Classification Review Dashboard")
    st.caption("Macro & specialist confusion matrices with per-sample drill-down (OCR text, image, reasoning, confidence).")

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload results CSV", type=["csv"])
    default_path = st.sidebar.text_input("...or a path on disk", value="")

    source = uploaded if uploaded is not None else (default_path or None)
    if source is None:
        st.info("Upload a CSV or provide a path in the sidebar to get started.")
        st.stop()

    df, has_ocr = load_data(source)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required column(s): {missing}")
        st.stop()

    if not has_ocr:
        st.sidebar.warning("No `ocr_text` column found -- showing placeholder text. Add the real column later and it'll appear automatically, no code changes needed.")

    st.sidebar.metric("Total rows", len(df))

    tab1, tab2 = st.tabs(["🗂️ Main Class (Macro)", "🔍 Subclass (Specialist)"])

    with tab1:
        render_matrix_tab(
            df, true_col="macro_gt", pred_col="macro_decision",
            reason_cols=["macro_reason"], conf_cols=["macro_confidence"],
            title="Macro: macro_gt vs macro_decision",
            key_prefix="main",
        )

    with tab2:
        render_matrix_tab(
            df, true_col="ground_truth", pred_col="final_subcategory",
            reason_cols=["specialist_reason"], conf_cols=["specialist_confidence"],
            title="Specialist: ground_truth vs final_subcategory",
            key_prefix="sub",
        )


if __name__ == "__main__":
    main()