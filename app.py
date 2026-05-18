# ============================================================
#  Financial Risk Analytics Dashboard
#  Project 4 — Ayush Gautam | github.com/Ayush-Gautam-016
#  Run with:  streamlit run app.py
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
# streamlit  → the web app framework
# pandas     → data manipulation (tables, filtering, grouping)
# numpy      → random number generation for synthetic data
# plotly     → interactive charts

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO


# ── PAGE CONFIG ──────────────────────────────────────────────
# Must be the FIRST streamlit command in the script.
# Sets the browser tab title, icon, and layout width.

st.set_page_config(
    page_title="Financial Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",                 # use full browser width
    initial_sidebar_state="expanded"
)


# ── CUSTOM CSS ───────────────────────────────────────────────
# Inject CSS to style metric cards and improve visual polish.
# st.markdown with unsafe_allow_html=True lets us write raw HTML/CSS.

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fb; }

    /* Top header bar */
    .dashboard-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .dashboard-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .dashboard-header p {
        margin: 0.3rem 0 0;
        color: rgba(255,255,255,0.65);
        font-size: 0.9rem;
    }

    /* KPI metric card */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        border-left: 4px solid #1a1a2e;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 0.3rem;
    }

    /* Section heading style */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #eee;
    }

    /* Risk badge colors */
    .badge-high   { color: #c94f3a; background: #fce8e5; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-medium { color: #b07010; background: #fef3dc; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }
    .badge-low    { color: #1f6b3e; background: #e0f5ea; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; }

    /* Hide the default streamlit top padding */
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — GENERATE SYNTHETIC DATA
#  We create 500 fake but realistic transactions.
#  np.random.seed(42) means every run produces the SAME data
#  (reproducible = good for demos and portfolios).
# ══════════════════════════════════════════════════════════════

@st.cache_data   # cache_data tells Streamlit: only run this ONCE, then cache the result.
                 # Without this, data would regenerate every time the user clicks anything.
def generate_data(n=500):
    np.random.seed(42)

    # --- Transaction amounts ---
    # Most transactions are small (£10–£500), a few are large (up to £5000).
    # We mix two distributions: normal small spends + a few large ones.
    amounts = np.concatenate([
        np.random.exponential(scale=150, size=int(n * 0.85)),  # regular spend
        np.random.exponential(scale=800, size=int(n * 0.15))   # large transactions
    ])
    amounts = np.clip(amounts, 5, 5000).round(2)

    # --- Fraud probability ---
    # Most transactions are safe (low probability).
    # A small fraction (≈10%) have high fraud probability.
    fraud_probs = np.concatenate([
        np.random.beta(a=1.5, b=10, size=int(n * 0.90)),  # mostly safe (0.0–0.3)
        np.random.beta(a=8,   b=2,  size=int(n * 0.10))   # suspicious  (0.6–1.0)
    ])
    fraud_probs = np.clip(fraud_probs, 0, 1).round(3)

    # --- Risk score ---
    # Combines fraud probability + transaction amount into a 0–100 business score.
    # Formula: 60% fraud signal + 40% amount signal (normalised).
    amount_norm  = (amounts - amounts.min()) / (amounts.max() - amounts.min())
    risk_scores  = (0.60 * fraud_probs + 0.40 * amount_norm) * 100
    risk_scores  = np.clip(risk_scores, 0, 100).round(1)

    # --- Customer segment ---
    # Based on risk score thresholds (business rule, not ML).
    segments = pd.cut(
        risk_scores,
        bins=[0, 30, 65, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    # --- Merchant category ---
    categories = np.random.choice(
        ["Retail", "Online", "Travel", "Dining", "ATM", "Utilities"],
        size=n,
        p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
    )

    # --- Transaction time (last 90 days) ---
    base_date = pd.Timestamp("2025-01-01")
    days_offset = np.random.randint(0, 90, size=n)
    dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offset]

    # --- Assemble into a DataFrame ---
    df = pd.DataFrame({
        "Transaction ID"    : [f"TXN-{10000 + i}" for i in range(n)],
        "Date"              : dates,
        "Amount (£)"        : amounts,
        "Merchant Category" : categories,
        "Fraud Probability" : fraud_probs,
        "Risk Score"        : risk_scores,
        "Customer Segment"  : segments.astype(str),
    })

    # Label a transaction as fraud if probability > 0.55
    df["Fraud Flag"] = df["Fraud Probability"].apply(
        lambda p: "🚨 Fraud" if p > 0.55 else "✅ Legit"
    )

    return df.sort_values("Date").reset_index(drop=True)


# Load data
df = generate_data()


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — SIDEBAR FILTERS
#  Sidebar = the left panel that collapses on mobile.
#  All filters here control what the main page displays.
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 Filters")
    st.markdown("---")

    # --- Risk Tier Filter ---
    # st.multiselect lets users pick multiple options from a list.
    # default=all tiers means everything shows on first load.
    selected_segments = st.multiselect(
        label="Customer Risk Tier",
        options=["Low Risk", "Medium Risk", "High Risk"],
        default=["Low Risk", "Medium Risk", "High Risk"]
    )

    # --- Fraud Probability Threshold ---
    # st.slider returns a tuple (min_val, max_val) when you pass a tuple as value.
    fraud_range = st.slider(
        label="Fraud Probability Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),    # show all by default
        step=0.01,
        format="%.2f"
    )

    # --- Transaction Amount Filter ---
    amount_max = st.slider(
        label="Max Transaction Amount (£)",
        min_value=int(df["Amount (£)"].min()),
        max_value=int(df["Amount (£)"].max()),
        value=int(df["Amount (£)"].max()),
        step=50
    )

    # --- Merchant Category Filter ---
    all_categories = sorted(df["Merchant Category"].unique())
    selected_cats = st.multiselect(
        label="Merchant Category",
        options=all_categories,
        default=all_categories
    )

    st.markdown("---")
    st.caption("Dashboard by **Ayush Gautam**")
    st.caption("github.com/Ayush-Gautam-016")


# ── APPLY FILTERS ────────────────────────────────────────────
# Filter the DataFrame based on what the user selected above.
# Each condition produces a boolean Series (True/False per row).
# We combine them with & (AND) to keep only rows matching ALL filters.

filtered_df = df[
    (df["Customer Segment"].isin(selected_segments)) &
    (df["Fraud Probability"] >= fraud_range[0]) &
    (df["Fraud Probability"] <= fraud_range[1]) &
    (df["Amount (£)"] <= amount_max) &
    (df["Merchant Category"].isin(selected_cats))
]


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="dashboard-header">
    <h1>📊 Financial Risk Analytics Dashboard</h1>
    <p>Real-time fraud detection · Customer risk segmentation · Transaction monitoring</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — KPI METRIC CARDS
#  st.columns(4) creates 4 equal-width columns side by side.
#  We put one metric card in each column.
# ══════════════════════════════════════════════════════════════

# Compute KPI values from the FILTERED data
total_txns       = len(filtered_df)
fraud_cases      = (filtered_df["Fraud Flag"] == "🚨 Fraud").sum()
avg_fraud_prob   = filtered_df["Fraud Probability"].mean()
high_risk_count  = (filtered_df["Customer Segment"] == "High Risk").sum()
fraud_rate_pct   = (fraud_cases / total_txns * 100) if total_txns > 0 else 0

# Display cards in 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{total_txns:,}</div>
        <div class="metric-sub">After filters applied</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #c94f3a;">
        <div class="metric-label">Fraud Cases Detected</div>
        <div class="metric-value" style="color:#c94f3a;">{fraud_cases:,}</div>
        <div class="metric-sub">{fraud_rate_pct:.1f}% of transactions</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #c8973a;">
        <div class="metric-label">Avg Fraud Probability</div>
        <div class="metric-value" style="color:#c8973a;">{avg_fraud_prob:.3f}</div>
        <div class="metric-sub">0 = safe · 1 = certain fraud</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #5c3d9e;">
        <div class="metric-label">High Risk Customers</div>
        <div class="metric-value" style="color:#5c3d9e;">{high_risk_count:,}</div>
        <div class="metric-sub">Risk score &gt; 65</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — CHARTS
#  We use Plotly Express (px) — the simplest way to make
#  interactive charts in Python. Each chart is one function call.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">📈 Analytics Charts</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

# ── CHART 1: Fraud Probability Distribution ───────────────────
# A histogram shows how fraud probabilities are spread.
# Most should cluster near 0 (safe). Suspicious ones appear near 1.

with chart_col1:
    fig1 = px.histogram(
        filtered_df,
        x="Fraud Probability",
        nbins=40,                       # number of bars
        color_discrete_sequence=["#1a1a2e"],
        title="Fraud Probability Distribution",
        labels={"Fraud Probability": "Fraud Probability Score"}
    )
    # Add a vertical red line at the 0.55 fraud threshold
    fig1.add_vline(
        x=0.55,
        line_dash="dash",
        line_color="#c94f3a",
        annotation_text="Fraud threshold (0.55)",
        annotation_position="top right"
    )
    fig1.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, b=20, l=20, r=20),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── CHART 2: Risk Score by Customer Segment ───────────────────
# Box plot shows the spread of risk scores within each segment.
# This validates that our segmentation logic makes sense.

with chart_col2:
    segment_order = ["Low Risk", "Medium Risk", "High Risk"]
    color_map = {
        "Low Risk"    : "#1f6b3e",
        "Medium Risk" : "#c8973a",
        "High Risk"   : "#c94f3a"
    }
    fig2 = px.box(
        filtered_df,
        x="Customer Segment",
        y="Risk Score",
        color="Customer Segment",
        color_discrete_map=color_map,
        category_orders={"Customer Segment": segment_order},
        title="Risk Score by Customer Segment",
    )
    fig2.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, b=20, l=20, r=20),
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── CHART 3 & 4: Second row of charts ────────────────────────

chart_col3, chart_col4 = st.columns(2)

# ── CHART 3: Transaction Amount vs Fraud Probability ─────────
# Scatter plot — each dot is one transaction.
# Colour = customer segment. Helps spot if large amounts = higher fraud.

with chart_col3:
    fig3 = px.scatter(
        filtered_df.sample(min(300, len(filtered_df)), random_state=1),  # sample for speed
        x="Amount (£)",
        y="Fraud Probability",
        color="Customer Segment",
        color_discrete_map=color_map,
        title="Transaction Amount vs Fraud Probability",
        opacity=0.65,
        hover_data=["Transaction ID", "Merchant Category", "Risk Score"]
    )
    fig3.add_hline(
        y=0.55,
        line_dash="dash",
        line_color="#c94f3a",
        annotation_text="Fraud threshold"
    )
    fig3.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, b=20, l=20, r=20),
        height=320
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── CHART 4: Fraud Cases by Merchant Category ─────────────────
# Bar chart showing which merchant categories have most fraud.
# Critical for fraud team investigations.

with chart_col4:
    fraud_by_cat = (
        filtered_df[filtered_df["Fraud Flag"] == "🚨 Fraud"]
        .groupby("Merchant Category")
        .size()
        .reset_index(name="Fraud Count")
        .sort_values("Fraud Count", ascending=True)
    )
    fig4 = px.bar(
        fraud_by_cat,
        x="Fraud Count",
        y="Merchant Category",
        orientation="h",            # horizontal bars
        title="Fraud Cases by Merchant Category",
        color="Fraud Count",
        color_continuous_scale=["#fce8e5", "#c94f3a"]
    )
    fig4.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, b=20, l=20, r=20),
        height=320,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — TRANSACTION DATA TABLE
#  Show the raw filtered transactions so analysts can inspect them.
#  st.dataframe() renders a scrollable, sortable table.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🧾 Transaction Records</div>', unsafe_allow_html=True)

# Format Amount column with £ sign for display
display_df = filtered_df.copy()
display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y")
display_df["Amount (£)"] = display_df["Amount (£)"].apply(lambda x: f"£{x:,.2f}")

# Show table — use_container_width stretches it to full column width
st.dataframe(
    display_df[[
        "Transaction ID", "Date", "Amount (£)",
        "Merchant Category", "Fraud Probability",
        "Risk Score", "Customer Segment", "Fraud Flag"
    ]],
    use_container_width=True,
    height=320
)

st.caption(f"Showing {len(filtered_df):,} of {len(df):,} transactions after filters.")


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — DOWNLOAD REPORT BUTTON
#  Converts the filtered DataFrame to CSV bytes, then offers
#  a download button. This is what makes the dashboard feel
#  like a real stakeholder tool — not just pretty charts.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">⬇️ Export Report</div>', unsafe_allow_html=True)

# Convert DataFrame to CSV stored in memory (not on disk)
# BytesIO = in-memory file — good for cloud deployments (no disk writes)
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")

col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])

with col_dl1:
    st.download_button(
        label="📥 Download CSV Report",
        data=csv_bytes,
        file_name="fraud_risk_report.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_dl2:
    # Show a summary stat for the export
    st.info(f"**{len(filtered_df):,} rows** · **{fraud_cases}** fraud cases · **{high_risk_count}** high risk", icon="📋")


# ══════════════════════════════════════════════════════════════
#  SECTION 8 — RISK INSIGHTS PANEL
#  A plain-English summary of what the filtered data shows.
#  This mimics what a risk analyst would write in a report.
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🧠 Risk Insights</div>', unsafe_allow_html=True)

ins1, ins2, ins3 = st.columns(3)

with ins1:
    # Top fraud merchant category
    if not filtered_df.empty:
        top_cat = (
            filtered_df[filtered_df["Fraud Flag"] == "🚨 Fraud"]
            .groupby("Merchant Category").size().idxmax()
            if fraud_cases > 0 else "N/A"
        )
        st.metric("Highest Fraud Category", top_cat, delta="Investigate first", delta_color="inverse")

with ins2:
    # Average amount in fraud transactions vs legit
    fraud_avg    = filtered_df[filtered_df["Fraud Flag"] == "🚨 Fraud"]["Amount (£)"].mean()
    legit_avg    = filtered_df[filtered_df["Fraud Flag"] == "✅ Legit"]["Amount (£)"].mean()
    delta_amount = fraud_avg - legit_avg if not np.isnan(fraud_avg) and not np.isnan(legit_avg) else 0
    st.metric(
        "Avg Fraud Transaction",
        f"£{fraud_avg:,.0f}" if not np.isnan(fraud_avg) else "N/A",
        delta=f"£{delta_amount:+,.0f} vs legit avg",
        delta_color="inverse"
    )

with ins3:
    # Fraud capture rate at 0.55 threshold
    true_fraud_in_data = (filtered_df["Fraud Probability"] > 0.55).sum()
    st.metric(
        "Cases Above Threshold",
        f"{true_fraud_in_data:,}",
        delta="at 0.55 cutoff",
        delta_color="off"
    )


# ── FOOTER ───────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center style='color:#aaa; font-size:0.8rem;'>"
    "Financial Risk Analytics Dashboard · Ayush Gautam · "
    "<a href='https://github.com/Ayush-Gautam-016' style='color:#aaa;'>GitHub</a> · "
    "<a href='https://linkedin.com/in/ayush-gautam-89098623b' style='color:#aaa;'>LinkedIn</a>"
    "</center>",
    unsafe_allow_html=True
)
