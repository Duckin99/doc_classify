"""
Multi-Label Classification Review Dashboard
=========================================
Run locally with:
    pip install streamlit pandas plotly
    streamlit run multi_label_review.py

Expects a CSV with (at minimum) these columns:
    filepath,
    xray_gt, ultrasound_gt, ecg_gt,
    contains_xray, contains_ultrasound, contains_ecg,
    xray_conf, ultrasound_conf, ecg_conf

If an `ocr_text` column is missing, a placeholder is automatically generated.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Multi-Label Review", layout="wide")

# Define the target labels we are evaluating
LABELS = ["xray", "ultrasound", "ecg"]

# Dynamically build the required columns based on the labels
REQUIRED_COLS = ["filepath"]
for label in LABELS:
    REQUIRED_COLS.extend([f"{label}_gt", f"contains_{label}", f"{label}_conf"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    has_ocr = "ocr_text" in df.columns
    if not has_ocr:
        df["ocr_text"] = df["filepath"].astype(str).apply(
            lambda fp: f"[OCR TEXT PLACEHOLDER -- column not wired up yet. File: {fp}]"
        )
    
    # Ensure boolean columns are strictly boolean/integers for plotting
    for label in LABELS:
        gt_col = f"{label}_gt"
        pred_col = f"contains_{label}"
        if gt_col in df.columns:
            df[gt_col] = df[gt_col].astype(bool)
        if pred_col in df.columns:
            df[pred_col] = df[pred_col].astype(bool)
            
    return df, has_ocr

def render_sample(row, idx):
    """Renders a single document sample with its multi-label predictions."""
    
    # Build a quick summary string for the header
    gt_tags = [L.upper() for L in LABELS if row.get(f"{L}_gt")]
    pred_tags = [L.upper() for L in LABELS if row.get(f"contains_{L}")]
    
    gt_str = ", ".join(gt_tags) if gt_tags else "NONE"
    pred_str = ", ".join(pred_tags) if pred_tags else "NONE"
    
    # Determine if it's a perfect match for visual cue
    is_perfect = set(gt_tags) == set(pred_tags)
    icon = "✅" if is_perfect else "⚠️"
    
    header = f"{icon} {Path(str(row.get('filepath', ''))).name}  —  GT: [{gt_str}]  →  Pred: [{pred_str}]"
    
    with st.expander(header):
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**Predictions vs Ground Truth**")
            # Create a clean display table for this specific document
            doc_stats = []
            for label in LABELS:
                gt_val = row.get(f"{label}_gt")
                pred_val = row.get(f"contains_{label}")
                conf = row.get(f"{pred_val}_confidence", 0.0)
                
                status = "Correct" if gt_val == pred_val else ("False Positive" if pred_val else "False Negative")
                
                doc_stats.append({
                    "Label": label.upper(),
                    "Ground Truth": "Yes" if gt_val else "No",
                    "Prediction": "Yes" if pred_val else "No",
                    "Confidence": f"{conf:.2f}%",
                    "Status": status
                })
            
            st.dataframe(pd.DataFrame(doc_stats), hide_index=True, use_container_width=True)

            st.markdown("**Document image**")
            fp = row.get("filepath", "")
            try:
                st.image(fp, use_container_width=True)
            except Exception:
                st.info(f"Could not load image from path:\n`{fp}`")

        with c2:
            st.markdown("**OCR text**")
            st.text_area(
                "ocr_text", value=str(row.get("ocr_text", "")),
                height=300, label_visibility="collapsed",
                key=f"ocr_{idx}",
            )


def render_performance_metrics(df: pd.DataFrame):
    """Calculates and displays global multi-label performance metrics."""
    st.subheader("Aggregate Performance Metrics")
    
    metrics = []
    perfect_match_count = 0
    
    # Calculate Exact Match Ratio (Subset Accuracy)
    for _, row in df.iterrows():
        is_exact = all(row[f"{L}_gt"] == row[f"contains_{L}"] for L in LABELS)
        if is_exact:
            perfect_match_count += 1
            
    exact_match_ratio = perfect_match_count / len(df) if len(df) > 0 else 0
    st.metric("Exact Match Ratio (All labels correct)", f"{exact_match_ratio:.2%}", help=f"{perfect_match_count} / {len(df)} documents")
    st.divider()
    
    # Calculate per-label metrics
    for label in LABELS:
        gt_col = f"{label}_gt"
        pred_col = f"contains_{label}"
        
        tp = ((df[gt_col] == True) & (df[pred_col] == True)).sum()
        fp = ((df[gt_col] == False) & (df[pred_col] == True)).sum()
        fn = ((df[gt_col] == True) & (df[pred_col] == False)).sum()
        tn = ((df[gt_col] == False) & (df[pred_col] == False)).sum()
        
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            "Modality": label.upper(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Support (GT Count)": int((df[gt_col] == True).sum())
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Render Bar Chart for Metrics
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=metrics_df['Modality'], y=metrics_df['Precision'], marker_color='#1f77b4'),
        go.Bar(name='Recall', x=metrics_df['Modality'], y=metrics_df['Recall'], marker_color='#ff7f0e'),
        go.Bar(name='F1 Score', x=metrics_df['Modality'], y=metrics_df['F1 Score'], marker_color='#2ca02c')
    ])
    fig.update_layout(barmode='group', title="Per-Label Precision, Recall, and F1", yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show underlying data table formatted nicely
    display_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

def render_drilldown_tab(df: pd.DataFrame):
    """Renders the filtering and sample drill-down section."""
    st.subheader("Filter and Review Samples")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        label_filter = st.selectbox("Select Target Label", ["All"] + [L.upper() for L in LABELS])
    with c2:
        condition_filter = st.selectbox("Filter Condition", ["All", "False Positives", "False Negatives", "Exact Matches", "Any Error"])
    with c3:
        sort_by_conf = st.checkbox("Sort by Lowest Confidence", value=False)

    filtered = df.copy()
    
    if label_filter != "All":
        target = label_filter.lower()
        if condition_filter == "False Positives":
            filtered = filtered[(filtered[f"{target}_gt"] == False) & (filtered[f"contains_{target}"] == True)]
        elif condition_filter == "False Negatives":
            filtered = filtered[(filtered[f"{target}_gt"] == True) & (filtered[f"contains_{target}"] == False)]
        elif condition_filter == "Exact Matches":
            filtered = filtered[filtered[f"{target}_gt"] == filtered[f"contains_{target}"]]
        elif condition_filter == "Any Error":
            filtered = filtered[filtered[f"{target}_gt"] != filtered[f"contains_{target}"]]
    else:
        # Global multi-label filtering
        if condition_filter == "Exact Matches":
            # Must be perfect across all labels
            condition = pd.Series([True] * len(filtered), index=filtered.index)
            for L in LABELS:
                condition = condition & (filtered[f"{L}_gt"] == filtered[f"contains_{L}"])
            filtered = filtered[condition]
        elif condition_filter in ["Any Error", "False Positives", "False Negatives"]:
            # Has at least one error across any label
            condition = pd.Series([False] * len(filtered), index=filtered.index)
            for L in LABELS:
                if condition_filter == "Any Error":
                    condition = condition | (filtered[f"{L}_gt"] != filtered[f"contains_{L}"])
                elif condition_filter == "False Positives":
                    condition = condition | ((filtered[f"{L}_gt"] == False) & (filtered[f"contains_{L}"] == True))
                elif condition_filter == "False Negatives":
                    condition = condition | ((filtered[f"{L}_gt"] == True) & (filtered[f"contains_{L}"] == False))
            filtered = filtered[condition]

    if sort_by_conf and label_filter != "All":
        filtered = filtered.sort_values(by=f"{label_filter.lower()}_conf", ascending=True)

    st.write(f"**{len(filtered)}** matching sample(s)")

    max_show = 200
    for i, (idx, row) in enumerate(filtered.iterrows()):
        if i >= max_show:
            st.info(f"Showing first {max_show} of {len(filtered)} -- narrow your filter to see more precisely.")
            break
        render_sample(row, idx)


def main():
    st.title("📄 Multi-Label Document Classification Review")
    st.caption("Review concurrent modality predictions (X-Ray, Ultrasound, ECG) with per-sample drill-down.")

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
        st.sidebar.warning("No `ocr_text` column found -- showing placeholder text.")

    st.sidebar.metric("Total documents", len(df))

    tab1, tab2 = st.tabs(["📊 Performance Metrics", "🔍 Sample Drill-down"])

    with tab1:
        render_performance_metrics(df)

    with tab2:
        render_drilldown_tab(df)

if __name__ == "__main__":
    main()