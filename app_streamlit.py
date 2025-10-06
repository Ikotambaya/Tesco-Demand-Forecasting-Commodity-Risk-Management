"""
Retail Demand & Commodity Risk Dashboard
📦 All data, models, and knowledge hosted on Hugging Face.
Repository: Uyane/tesco-project

Run with:
    streamlit run app_streamlit.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download
import altair as alt

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
DATA_REPO_ID = "Uyane/tesco-project"
DATA_REPO_TYPE = "dataset"  # everything is in the dataset repo

st.set_page_config(page_title="Retail Demand & Commodity Risk Dashboard", layout="wide")
st.title("📊 Retail Demand & Commodity Risk Dashboard")

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_data
def load_csv(filename, parse_dates=None):
    """Load CSV from Hugging Face dataset repo."""
    try:
        path = hf_hub_download(
            repo_id=DATA_REPO_ID,
            repo_type=DATA_REPO_TYPE,
            filename=filename,
            token=st.secrets["HF_TOKEN"]
        )
        df = pd.read_csv(path, parse_dates=parse_dates)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load {filename}: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(filename):
    """Load Joblib model from Hugging Face dataset repo."""
    try:
        path = hf_hub_download(
            repo_id=DATA_REPO_ID,      # use the dataset repo
            repo_type=DATA_REPO_TYPE,  # dataset type
            filename=filename,          # e.g., 'models/rf_store_nextday.joblib'
            token=st.secrets["HF_TOKEN"]
        )
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model {filename}: {e}")
        return None

# -----------------------------
# LOAD ALL DATASETS
# -----------------------------
with st.spinner("📡 Fetching datasets from Hugging Face..."):
    stores = load_csv("data/stores.csv")
    skus = load_csv("data/skus.csv")
    sales = load_csv("data/sales_data.csv", parse_dates=["date"])
    agg = load_csv("data/daily_store_agg.csv", parse_dates=["date"])
    commodities = load_csv("data/commodities.csv", parse_dates=["date"])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔎 Controls")
store_choice = st.sidebar.selectbox("Select Store", stores['store_id'].tolist() if not stores.empty else [])
view_choice = st.sidebar.radio(
    "Choose View",
    [
        "Store Insights",
        "Top SKUs",
        "Sales Insights",
        "Commodity Insights",
        "Forecasts",
        "Hedging Simulator",
        "RAG + LLM Explainer"
    ],
)

# -----------------------------
# STORE INSIGHTS
# -----------------------------
if view_choice == "Store Insights" and not agg.empty:
    st.subheader(f"📈 Store Demand Insights — {store_choice}")
    years = st.slider("Years to display", 1, 30, 15)
    end_date = agg['date'].max()
    start_date = end_date - pd.DateOffset(years=years)
    df = agg[(agg['store_id']==store_choice) & (agg['date']>=start_date)]

    if df.empty:
        st.warning("No data available for this store and period.")
    else:
        total_units = df['units_sold'].sum()
        avg_price = df['price'].mean()
        promo_days = df['on_promo'].sum()
        stockouts = df['stockout'].sum()
        total_revenue = (df['units_sold'] * df['price']).sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Units Sold", f"{total_units:,.0f}")
        c2.metric("Avg. Price", f"{avg_price:,.2f}")
        c3.metric("Promo Days", f"{promo_days:,}")
        c4.metric("Stockouts", f"{stockouts:,}")
        c5.metric("Total Revenue", f"${total_revenue:,.0f}")

        yearly = (
            df.groupby(df['date'].dt.year)['units_sold']
              .sum()
              .reset_index(name='units_sold')
              .rename(columns={'date':'year'})
        )
        yearly['year'] = pd.to_datetime(yearly['year'].astype(str) + "-01-01")
        st.line_chart(yearly.set_index('year')['units_sold'])
        st.caption("Yearly total units sold (1995–2025)")
        st.write("Recent daily data:")
        st.dataframe(df.sort_values('date', ascending=False).head(15))

# -----------------------------
# TOP SKUs
# -----------------------------
elif view_choice == "Top SKUs" and not sales.empty:
    st.subheader(f"🏆 Top SKUs by volume — {store_choice}")
    df_store = sales[sales['store_id']==store_choice]
    top_skus = df_store.groupby('sku_id')['units_sold'].sum().sort_values(ascending=False).head(20)
    st.bar_chart(top_skus)
    st.write("SKU details:", skus[skus['sku_id'].isin(top_skus.index)])

# -----------------------------
# SALES INSIGHTS
# -----------------------------
elif view_choice == "Sales Insights" and not sales.empty:
    st.subheader(f"📊 Promotion Impact — {store_choice}")
    df_store = sales[sales['store_id']==store_choice]
    categories = ["All"] + skus['category'].dropna().unique().tolist()
    cat_choice = st.selectbox("Filter by Category", categories)
    df_filtered = df_store.copy() if cat_choice=="All" else df_store[df_store['sku_id'].isin(skus[skus['category']==cat_choice]['sku_id'])]

    promo_stats = df_filtered.groupby('on_promo')['units_sold'].mean().reset_index()
    promo_stats['promo_label'] = promo_stats['on_promo'].map({0:'No Promo',1:'Promo'})
    st.write("Average units sold:")
    chart = alt.Chart(promo_stats).mark_bar().encode(
        x='promo_label:N',
        y='units_sold:Q',
        color='promo_label:N',
        tooltip=['promo_label','units_sold']
    ).properties(width=400,height=300,title=f"Average Units Sold: Promo vs No Promo ({cat_choice})")
    st.altair_chart(chart, use_container_width=True)

    try:
        lift_pct = ((promo_stats.loc[promo_stats['on_promo']==1,'units_sold'].values[0] -
                     promo_stats.loc[promo_stats['on_promo']==0,'units_sold'].values[0]) /
                     promo_stats.loc[promo_stats['on_promo']==0,'units_sold'].values[0]) * 100
    except IndexError:
        lift_pct = 0
    st.write(f"**Promotion Lift:** {lift_pct:.1f}%")

# -----------------------------
# COMMODITY INSIGHTS
# -----------------------------
elif view_choice == "Commodity Insights" and not commodities.empty:
    st.subheader("🌾 Commodity Market Trends")
    com_choice = st.selectbox("Commodity", ['wheat_spot','dairy_spot','oilseed_spot'])
    years = st.slider("Years to display", 1, 30, 5)
    end_date = commodities['date'].max()
    start_date = end_date - pd.DateOffset(years=years)
    com_series = commodities.set_index('date')[com_choice]
    st.line_chart(com_series.loc[start_date:end_date])
    st.write("Recent values:", com_series.tail(10))

    vol = com_series.pct_change().rolling(90).std()
    st.line_chart(vol.loc[start_date:end_date])
    st.caption("Rolling 90-day volatility")

    recent = commodities[commodities['date']>=end_date - pd.DateOffset(years=3)]
    corr = recent[['wheat_spot','dairy_spot','oilseed_spot']].pct_change().corr()
    st.subheader("Commodity Correlation Matrix (last 3 years)")
    st.dataframe(corr)

# -----------------------------
# FORECASTS
# -----------------------------
elif view_choice == "Forecasts":
    st.subheader("🔮 ML Forecasts")
    tab1, tab2 = st.tabs(["Store Demand Forecast", "Commodity Forecast"])

    # Store Forecast
    with tab1:
        model_store = load_model("models/rf_store_nextday.joblib")
        if model_store is not None and not agg.empty:
            df_store = agg[agg['store_id']==store_choice].copy().sort_values('date')
            df_store['dow'] = df_store['date'].dt.weekday
            for lag in [1,7,14]:
                df_store[f'units_lag_{lag}'] = df_store['units_sold'].shift(lag)
            df_store['ma_7'] = df_store['units_sold'].rolling(7, min_periods=1).mean()
            df_store = df_store.dropna()
            feats = ['units_lag_1','units_lag_7','units_lag_14','ma_7','on_promo','stockout','price','avg_temp','dow']
            y_pred = model_store.predict(df_store[feats].tail(30))
            st.line_chart(pd.Series(y_pred, index=df_store['date'].tail(30), name="Forecasted Units"))
        else:
            st.error("Store forecast model not available")

    # Commodity Forecast
    with tab2:
        model_comm = load_model("models/rf_wheat_nextday.joblib")
        if model_comm is not None and not commodities.empty:
            df_comm = commodities.copy().sort_values('date')
            df_comm['wheat_lag_1'] = df_comm['wheat_spot'].shift(1)
            df_comm['wheat_lag_7'] = df_comm['wheat_spot'].shift(7)
            df_comm['wheat_ma_7'] = df_comm['wheat_spot'].rolling(7, min_periods=1).mean()
            df_comm = df_comm.dropna()
            y_pred = model_comm.predict(df_comm[['wheat_lag_1','wheat_lag_7','wheat_ma_7']].tail(30))
            st.line_chart(pd.Series(y_pred, index=df_comm['date'].tail(30), name="Wheat Forecast"))
        else:
            st.error("Commodity forecast model not available")

# -----------------------------
# HEDGING SIMULATOR
# -----------------------------
elif view_choice == "Hedging Simulator":
    st.subheader("🛡️ Hedging Simulation")

    # Simulation controls
    notional = st.number_input("Notional GBP exposure", value=1_000_000, step=50_000)
    days = st.slider("Simulation horizon (days)", min_value=30, max_value=180, value=90)
    n_sims = st.slider("Number of simulations", min_value=200, max_value=2000, value=500)

    # Initialize session state
    if "agg_pnls" not in st.session_state:
        st.session_state.agg_pnls = None
    if "sims_dict" not in st.session_state:
        st.session_state.sims_dict = {}

    run_sim = st.button("Run Simulation")

    # Run simulation if button clicked or previous results exist
    if run_sim or st.session_state.agg_pnls is not None:
        if run_sim or st.session_state.agg_pnls is None:
            from hedging_simulator import load_commodities_safe, residual_bootstrap_sim, compute_pnl_for_basket

            # Load commodity data
            comm = load_commodities_safe()

            # Define basket
            basket = {
                "wheat_spot": {"share": 0.6},
                "dairy_spot": {"share": 0.3},
                "oilseed_spot": {"share": 0.1}
            }

            sims_dict = {}
            agg_pnls = None

            # Run simulations per commodity
            for col in basket:
                sims = residual_bootstrap_sim(comm[col], n_days=days, n_sims=n_sims)
                sims_dict[col] = sims
                basket[col]["last_price"] = comm[col].iloc[-1]

            # Compute aggregated P&L
            for col, info in basket.items():
                pnl = compute_pnl_for_basket(sims_dict[col], info, notional)
                agg_pnls = pnl if agg_pnls is None else agg_pnls + pnl

            # Save results to session state
            st.session_state.agg_pnls = agg_pnls
            st.session_state.sims_dict = sims_dict

        # Use session_state results for display
        agg_pnls = st.session_state.agg_pnls
        sims_dict = st.session_state.sims_dict

        # Summarize and display
        df_summary = pd.DataFrame({
            "mean": agg_pnls.mean(axis=0),
            "p5": np.percentile(agg_pnls, 5, axis=0),
            "p50": np.percentile(agg_pnls, 50, axis=0),
            "p95": np.percentile(agg_pnls, 95, axis=0),
        }, index=pd.RangeIndex(1, agg_pnls.shape[1] + 1))  # Day index

        st.line_chart(df_summary[['p5', 'p50', 'p95']])
        st.write("Summary at key horizons:")
        for d in [29, 59, min(89, days - 1)]:
            st.write(f"Day {d+1}: mean {df_summary['mean'].iloc[d]:,.0f}, "
                     f"p5 {df_summary['p5'].iloc[d]:,.0f}, "
                     f"p95 {df_summary['p95'].iloc[d]:,.0f}")

        # Display top scenarios for each commodity
        st.subheader("Top 5 simulated scenarios per commodity (first 10 days)")
        for com, sims in sims_dict.items():
            st.write(f"**{com}**")
            st.dataframe(pd.DataFrame(
                sims[:5, :10],
                columns=[f"Day {i+1}" for i in range(10)]
            ))

        st.success("Simulation complete ✅")

# -----------------------------
# RAG + LLM Explainer
# -----------------------------
elif view_choice == "RAG + LLM Explainer":
    st.subheader("💬 Ask Questions About Store S01")
    from rag_llm_explainer import build_knowledge_base, retrieve, call_openai_with_context

    kb_path = hf_hub_download(DATA_REPO_ID, repo_type=DATA_REPO_TYPE, filename="knowledge/kb.csv", token=st.secrets["HF_TOKEN"])
    build_knowledge_base()  # ensure KB exists

    query = st.text_input("Enter your question about Store S01:", value="Why did store S01 see a drop in units last week?")
    if st.button("Get Answer") and query.strip():
        context = retrieve(query)
        if context:
            answer = call_openai_with_context(query, context)
            st.write("**Retrieved Context:**")
            for c in context:
                st.write(f"- {c}")
            st.write("**LLM Answer / Recommendation:**")
            st.success(answer)
        else:
            st.warning("No relevant context found in KB.")

