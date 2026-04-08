"""
Invoice Anomaly Detector — Streamlit Web App
=============================================
Run with:  streamlit run app.py
"""

import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Import from the main module
sys.path.insert(0, os.path.dirname(__file__))
from invoice_detector import (
    InvoiceAnomalyDetector,
    generate_excel_report,
    generate_invoice_data,
)

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #2F75B5;
    }
    .risk-high   { color: #E74C3C; font-weight: bold; }
    .risk-medium { color: #F39C12; font-weight: bold; }
    .risk-low    { color: #F1C40F; font-weight: bold; }
    .stDataFrame { font-size: 13px; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-in-list.png", width=64)
    st.title("Invoice Anomaly\nDetector")
    st.caption("Consulting Portfolio Project")
    st.divider()

    st.subheader("Data Source")
    data_mode = st.radio(
        "Choose input",
        ["Use sample data", "Upload CSV"],
        index=0,
    )

    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload invoice CSV",
            type=["csv"],
            help="Required columns: invoice_id, date, vendor, category, amount, cost_center, approved_by",
        )

    st.divider()
    st.subheader("Detection Settings")
    contamination = st.slider(
        "Expected anomaly rate",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%0.0f%%",
        help="IsolationForest contamination parameter. Set to your estimated fraud rate.",
    )

    n_normal = st.slider("Sample size (normal invoices)", 100, 1000, 500, 50,
                         disabled=(data_mode == "Upload CSV"))

    st.divider()
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    st.caption("v1.0 | github.com/yourhandle")


# ── Main area ────────────────────────────────────────────────────
st.title("Invoice Anomaly Detector")
st.markdown("Automated multi-layer anomaly detection for financial invoice data — combining rule-based checks with Machine Learning (IsolationForest).")

if not run_button:
    # Landing state
    st.info("Configure the settings in the sidebar and click **Run Analysis** to start.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Detection Module 1**\n\nDuplicate Invoice IDs\n\n`Rule-based` `HIGH risk`")
    with col2:
        st.markdown("**Detection Module 2**\n\nWeekend Bookings\n\n`Rule-based` `MEDIUM risk`")
    with col3:
        st.markdown("**Detection Module 3**\n\nML Amount Outliers\n\n`IsolationForest` `HIGH risk`")
    with col4:
        st.markdown("**Detection Module 4**\n\nRound Number Bias\n\n`Rule-based` `LOW risk`")
    st.stop()


# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    if data_mode == "Upload CSV" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            st.success(f"Loaded {len(df):,} invoices from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()
    else:
        df = generate_invoice_data(n_normal=n_normal, n_anomalies=30)
        st.info(f"Using generated sample data ({len(df):,} invoices with embedded anomalies)")


# ── Run detection ────────────────────────────────────────────────
with st.spinner("Running anomaly detection..."):
    detector = InvoiceAnomalyDetector(contamination=contamination)
    results = detector.run(df)
    summary = results["summary"]
    anomalies = results["all_anomalies"]


# ── KPI Row ──────────────────────────────────────────────────────
st.divider()
st.subheader("Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Invoices Analyzed",  f"{summary['total_invoices']:,}")
col2.metric("Anomalies Found",    f"{summary['total_anomalies']:,}",
            delta=f"{summary['anomaly_rate_pct']}% of total", delta_color="inverse")
col3.metric("HIGH Risk",          f"{summary['high_risk_count']:,}", delta_color="inverse")
col4.metric("Total Volume",       f"EUR {summary['total_volume_eur']/1e6:.1f}M")
col5.metric("Value at Risk",      f"EUR {summary['anomaly_value_eur']/1e6:.1f}M",
            delta_color="inverse")


# ── Charts ───────────────────────────────────────────────────────
st.divider()
st.subheader("Analysis Dashboard")

chart_col1, chart_col2, chart_col3 = st.columns(3)

# Chart 1: Anomaly breakdown
with chart_col1:
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    breakdown = summary["breakdown"]
    colors = ["#E74C3C", "#E67E22", "#C0392B", "#F1C40F"]
    bars = ax1.bar(
        [k.replace(" ", "\n") for k in breakdown.keys()],
        breakdown.values(),
        color=colors, edgecolor="white", linewidth=1.5,
    )
    for bar, val in zip(bars, breakdown.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_title("Anomalies by Type", fontweight="bold", pad=10)
    ax1.set_ylabel("Count")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=8)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# Chart 2: Pie chart
with chart_col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    normal_count = summary["total_invoices"] - summary["total_anomalies"]
    ax2.pie(
        [normal_count, summary["total_anomalies"]],
        labels=["Normal", "Anomaly"],
        colors=["#2ECC71", "#E74C3C"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    ax2.set_title("Invoice Health", fontweight="bold", pad=10)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# Chart 3: Risk levels
with chart_col3:
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    risk_counts = anomalies["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
    risk_colors = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#F1C40F"}
    bars3 = ax3.bar(
        risk_counts.index,
        risk_counts.values,
        color=[risk_colors[r] for r in risk_counts.index],
        edgecolor="white", linewidth=1.5,
    )
    for bar, val in zip(bars3, risk_counts.values):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     int(val), ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax3.set_title("By Risk Level", fontweight="bold", pad=10)
    ax3.set_ylabel("Count")
    ax3.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ── Anomaly Table ────────────────────────────────────────────────
st.divider()
st.subheader("Flagged Invoices")

# Filter controls
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    risk_filter = st.multiselect(
        "Risk Level",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )
with filter_col2:
    type_filter = st.multiselect(
        "Anomaly Type",
        options=list(summary["breakdown"].keys()),
        default=list(summary["breakdown"].keys()),
    )
with filter_col3:
    sort_by = st.selectbox("Sort by", ["risk_level", "amount", "date", "vendor"], index=1)

# Apply filters
display_df = anomalies[
    anomalies["risk_level"].isin(risk_filter) &
    anomalies["detected_anomaly"].isin(type_filter)
].copy()

display_df = display_df.sort_values(sort_by, ascending=(sort_by not in ["amount"]))

# Format for display
display_cols = ["invoice_id", "date", "vendor", "category", "amount",
                "detected_anomaly", "risk_level", "recommendation"]
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df["amount"] = display_df["amount"].apply(lambda x: f"EUR {x:,.2f}")

# Color-code risk level
def highlight_risk(row):
    colors = {"HIGH": "background-color: #FFE5E5", "MEDIUM": "background-color: #FFF3E0", "LOW": "background-color: #FFFDE7"}
    return [colors.get(row["risk_level"], "") if col == "risk_level" else "" for col in row.index]

st.dataframe(
    display_df[display_cols].style.apply(highlight_risk, axis=1),
    use_container_width=True,
    height=400,
    hide_index=True,
)

st.caption(f"Showing {len(display_df)} of {len(anomalies)} flagged invoices")


# ── Download buttons ─────────────────────────────────────────────
st.divider()
st.subheader("Export")

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    # Excel download
    excel_path = "output/anomaly_report.xlsx"
    os.makedirs("output", exist_ok=True)
    generate_excel_report(results, excel_path)
    with open(excel_path, "rb") as f:
        st.download_button(
            label="Download Excel Report",
            data=f,
            file_name=f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )

with dl_col2:
    # CSV download
    csv_data = anomalies[display_cols].copy()
    csv_data["date"] = csv_data["date"]  # already formatted
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Anomalies CSV",
        data=csv_buffer.getvalue(),
        file_name=f"flagged_invoices_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("Invoice Anomaly Detector | IT Consulting Portfolio | Built with Python + Streamlit")
