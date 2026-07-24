"""
Multi-Label Classification Review Dashboard
=========================================
Run locally with:
    pip install streamlit pandas plotly
    streamlit run multi_label_review.py

Expects a CSV with (at minimum) these columns:
    filepath, flags_reason,
    xray_gt, ultrasound_gt, ecg_gt,
    contains_xray, contains_ultrasound, contains_ecg,
    contains_xray_confidence, contains_ultrasound_confidence, contains_ecg_confidence

If an `ocr_text` column is missing, a placeholder is automatically generated.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Multi-Label Review", layout="wide")

# Define the target labels we are evaluating
LABELS = ["xray", "ultrasound", "ecg"]

# Dynamically build the required columns based on the labels
REQUIRED_COLS = ["filepath", "flags_reason"]
for label in LABELS:
    REQUIRED_COLS.extend([f"{label}_gt", f"contains_{label}", f"contains_{label}_confidence"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    has_ocr = "ocr_text" in df.columns
    if not has_ocr:
        df["ocr_text"] = df["filepath"].astype(str).apply(
            lambda fp: f"[OCR TEXT PLACEHOLDER -- column not wired up yet. File: {fp}]"
        )
    
    # Ensure boolean columns are strictly boolean for calculations
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
    
    gt_tags = [L.upper() for L in LABELS if row.get(f"{L}_gt")]
    pred_tags = [L.upper() for L in LABELS if row.get(f"contains_{L}")]
    
    gt_str = ", ".join(gt_tags) if gt_tags else "NONE"
    pred_str = ", ".join(pred_tags) if pred_tags else "NONE"
    
    is_perfect = set(gt_tags) == set(pred_tags)
    icon = "✅" if is_perfect else "⚠️"
    
    header = f"{icon} {Path(str(row.get('filepath', ''))).name}  —  GT: [{gt_str}]  →  Pred: [{pred_str}]"
    
    with st.expander(header):
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**Predictions vs Ground Truth**")
            doc_stats = []
            for label in LABELS:
                gt_val = row.get(f"{label}_gt")
                pred_val = row.get(f"contains_{label}")
                conf = row.get(f"contains_{label}_confidence", 0.0)
                
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
            st.markdown("**Reasoning (flags_reason)**")
            st.info(row.get("flags_reason", "No reasoning provided."))
            
            st.markdown("**OCR text**")
            st.text_area(
                "ocr_text", value=str(row.get("ocr_text", "")),
                height=300, label_visibility="collapsed",
                key=f"ocr_{idx}",
            )


def render_performance_metrics(df: pd.DataFrame):
    """Calculates and displays global multi-label performance metrics and visualizations."""
    st.subheader("Aggregate Performance Metrics")
    
    # 1. Exact Match Ratio
    perfect_match_count = sum(all(row[f"{L}_gt"] == row[f"contains_{L}"] for L in LABELS) for _, row in df.iterrows())
    exact_match_ratio = perfect_match_count / len(df) if len(df) > 0 else 0
    st.metric("Exact Match Ratio (All labels correct)", f"{exact_match_ratio:.2%}", help=f"{perfect_match_count} / {len(df)} documents")
    st.divider()
    
    # 2. Per-Label Metrics Calculation
    metrics = []
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
            "Support (GT Count)": int((df[gt_col] == True).sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Render Bar Chart for Metrics (Full Width)
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=metrics_df['Modality'], y=metrics_df['Precision'], 
               marker_color='#5C6BC0', text=metrics_df['Precision'], texttemplate='%{text:.1%}', textposition='outside'),
        go.Bar(name='Recall', x=metrics_df['Modality'], y=metrics_df['Recall'], 
               marker_color='#7E57C2', text=metrics_df['Recall'], texttemplate='%{text:.1%}', textposition='outside'),
        go.Bar(name='F1 Score', x=metrics_df['Modality'], y=metrics_df['F1 Score'], 
               marker_color='#3F51B5', text=metrics_df['F1 Score'], texttemplate='%{text:.1%}', textposition='outside')
    ])
    fig.update_layout(
        barmode='group', 
        title="Per-Label Precision, Recall, and F1", 
        yaxis_tickformat='.0%',
        yaxis_range=[0, 1.15], # Extend slightly so outside text fits
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Add subtle y-axis grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show underlying data table formatted nicely (Full Width)
    display_df = metrics_df.drop(columns=['TP', 'FP', 'FN', 'TN']).copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # 5. Individual 2x2 Confusion Matrices
    st.subheader("Per-Label Confusion Matrices")
    cols = st.columns(len(LABELS))
    for i, label in enumerate(LABELS):
        with cols[i]:
            m = metrics_df[metrics_df['Modality'] == label.upper()].iloc[0]
            z = [[m['TN'], m['FP']], [m['FN'], m['TP']]]
            fig_cm = px.imshow(
                z,
                x=['Pred: False', 'Pred: True'],
                y=['GT: False', 'GT: True'],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"{label.upper()}"
            )
            fig_cm.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_cm, use_container_width=True)


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
            condition = pd.Series([True] * len(filtered), index=filtered.index)
            for L in LABELS:
                condition = condition & (filtered[f"{L}_gt"] == filtered[f"contains_{L}"])
            filtered = filtered[condition]
        elif condition_filter in ["Any Error", "False Positives", "False Negatives"]:
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
        filtered = filtered.sort_values(by=f"contains_{label_filter.lower()}_confidence", ascending=True)

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