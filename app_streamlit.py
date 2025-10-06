# app_streamlit.py
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
from typing import Dict, Any

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------

DATA_REPO_ID = "Uyane/tesco-project"
DATA_REPO_TYPE = "dataset"  # everything is in the dataset repo

# -----------------------------
# DIRECTORY PATHS (Hugging Face structure)
# -----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
KNOW_DIR = "knowledge"
SIM_DIR = "sim_results"


# optional HF token from Streamlit secrets (if repo is private)
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

st.set_page_config(page_title="Retail Demand & Commodity Risk Dashboard", layout="wide")
st.title("📊 Retail Demand & Commodity Risk Dashboard")

# -----------------------------
# HELPERS: HF downloads + local cache
# -----------------------------
@st.cache_data
def hf_download_to_path(filename: str) -> str:
    """
    Download a file from the dataset repo to local cache and return local path.
    Uses token if present.
    """
    try:
        path = hf_hub_download(
            repo_id=DATA_REPO_ID,
            repo_type=DATA_REPO_TYPE,
            filename=filename,
            token=HF_TOKEN,
        )
        return path
    except Exception as e:
        raise

@st.cache_data
def load_csv(filename: str, parse_dates=None) -> pd.DataFrame:
    """Load CSV from Hugging Face dataset repo (returns DataFrame)."""
    try:
        local_path = hf_download_to_path(filename)
        df = pd.read_csv(local_path, parse_dates=parse_dates)
        return df
    except Exception as e:
        # Bubble a friendly error message to UI rather than stopping execution
        st.error(f"❌ Failed to load {filename}: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(filename: str):
    """Load joblib model from Hugging Face dataset repo (returns loaded model or None)."""
    try:
        local_path = hf_download_to_path(filename)
        model = joblib.load(local_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model {filename}: {e}")
        return None

# -----------------------------
# Lightweight hedging helper functions (embedded to avoid import issues)
# -----------------------------
def residual_bootstrap_sim(series: pd.Series, n_days: int = 90, n_sims: int = 2000) -> np.ndarray:
    """
    Simulate future price paths using AR(1)-like residual bootstrapping.
    Returns array shape (n_sims, n_days).
    """
    s = series.dropna().reset_index(drop=True)
    if s.empty:
        raise ValueError("Input series is empty")
    diffs = s.diff().dropna()
    if diffs.empty:
        raise ValueError("Series too short for differencing")
    mu = diffs.mean()
    resid = diffs - mu
    sims = np.zeros((n_sims, n_days))
    last = s.iloc[-1]
    for i in range(n_sims):
        path = last
        row = []
        for d in range(n_days):
            e = np.random.choice(resid.values)
            nxt = path + mu + e
            row.append(nxt)
            path = nxt
        sims[i, :] = np.array(row)
    return sims

def compute_pnl_for_basket(sims: np.ndarray, last_price: float, share: float, notional: float) -> np.ndarray:
    """
    Compute P&L matrix for given sims (n_sims x n_days)
    each scenario: pct change = (price - last) / last; pnl = pct_change * (notional * share)
    """
    pct_changes = (sims - last_price) / last_price
    pnl = pct_changes * (notional * share)
    return pnl

# -----------------------------
# LOAD ALL DATASETS (from HF)
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
    df = agg[(agg['store_id'] == store_choice) & (agg['date'] >= start_date)]

    if df.empty:
        st.warning("No data available for this store and period.")
    else:
        total_units = df['units_sold'].sum()
        avg_price = df['price'].mean()
        promo_days = int(df['on_promo'].sum())
        stockouts = int(df['stockout'].sum())
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
        # fix: yearly['year'] based on grouped column name
        yearly['year'] = pd.to_datetime(yearly['year'].astype(str) + "-01-01")
        st.line_chart(yearly.set_index('year')['units_sold'])
        st.caption("Yearly total units sold")
        st.write("Recent daily data:")
        st.dataframe(df.sort_values('date', ascending=False).head(15))

# -----------------------------
# TOP SKUs
# -----------------------------
elif view_choice == "Top SKUs" and not sales.empty:
    st.subheader(f"🏆 Top SKUs by volume — {store_choice}")
    df_store = sales[sales['store_id'] == store_choice] if not sales.empty and store_choice else pd.DataFrame()
    top_skus = df_store.groupby('sku_id')['units_sold'].sum().sort_values(ascending=False).head(20) if not df_store.empty else pd.Series()
    st.bar_chart(top_skus)
    if not skus.empty:
        st.write("SKU details:")
        st.dataframe(skus[skus['sku_id'].isin(top_skus.index)])

# -----------------------------
# SALES INSIGHTS
# -----------------------------
elif view_choice == "Sales Insights" and not sales.empty:
    st.subheader(f"📊 Promotion Impact — {store_choice}")
    df_store = sales[sales['store_id'] == store_choice] if not sales.empty and store_choice else pd.DataFrame()
    categories = ["All"] + (skus['category'].dropna().unique().tolist() if not skus.empty else [])
    cat_choice = st.selectbox("Filter by Category", categories)
    if cat_choice == "All":
        df_filtered = df_store.copy()
    else:
        sku_ids = skus[skus['category'] == cat_choice]['sku_id'] if not skus.empty else []
        df_filtered = df_store[df_store['sku_id'].isin(sku_ids)]

    if df_filtered.empty:
        st.info("No sales data for this selection.")
    else:
        promo_stats = df_filtered.groupby('on_promo')['units_sold'].mean().reset_index()
        promo_stats['promo_label'] = promo_stats['on_promo'].map({0: 'No Promo', 1: 'Promo'})
        st.write("Average units sold:")
        chart = alt.Chart(promo_stats).mark_bar().encode(
            x='promo_label:N',
            y='units_sold:Q',
            color='promo_label:N',
            tooltip=['promo_label','units_sold']
        ).properties(width=400, height=300, title=f"Average Units Sold: Promo vs No Promo ({cat_choice})")
        st.altair_chart(chart, use_container_width=True)

        try:
            no_promo = promo_stats.loc[promo_stats['on_promo'] == 0, 'units_sold'].values[0]
            promo = promo_stats.loc[promo_stats['on_promo'] == 1, 'units_sold'].values[0]
            lift_pct = ((promo - no_promo) / no_promo) * 100
        except Exception:
            lift_pct = 0.0
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

    recent = commodities[commodities['date'] >= end_date - pd.DateOffset(years=3)]
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
        if model_store is not None and not agg.empty and store_choice:
            df_store = agg[agg['store_id'] == store_choice].copy().sort_values('date')
            df_store['dow'] = df_store['date'].dt.weekday
            for lag in [1,7,14]:
                df_store[f'units_lag_{lag}'] = df_store['units_sold'].shift(lag)
            df_store['ma_7'] = df_store['units_sold'].rolling(7, min_periods=1).mean()
            df_store = df_store.dropna()
            feats = ['units_lag_1','units_lag_7','units_lag_14','ma_7','on_promo','stockout','price','avg_temp','dow']
            try:
                y_pred = model_store.predict(df_store[feats].tail(30))
                st.line_chart(pd.Series(y_pred, index=df_store['date'].tail(30), name="Forecasted Units"))
            except Exception as e:
                st.error(f"Store forecast error: {e}")
        else:
            st.error("Store forecast model not available or insufficient data.")

    # Commodity Forecast
    with tab2:
        model_comm = load_model("models/rf_wheat_nextday.joblib")
        if model_comm is not None and not commodities.empty:
            df_comm = commodities.copy().sort_values('date')
            df_comm['wheat_lag_1'] = df_comm['wheat_spot'].shift(1)
            df_comm['wheat_lag_7'] = df_comm['wheat_spot'].shift(7)
            df_comm['wheat_ma_7'] = df_comm['wheat_spot'].rolling(7, min_periods=1).mean()
            df_comm = df_comm.dropna()
            try:
                y_pred = model_comm.predict(df_comm[['wheat_lag_1','wheat_lag_7','wheat_ma_7']].tail(30))
                st.line_chart(pd.Series(y_pred, index=df_comm['date'].tail(30), name="Wheat Forecast"))
            except Exception as e:
                st.error(f"Commodity forecast error: {e}")
        else:
            st.error("Commodity forecast model not available or insufficient data.")

# -----------------------------
# HEDGING SIMULATOR (precomputed + run new)
# -----------------------------
elif view_choice == "Hedging Simulator":
    st.subheader("🛡️ Hedging Simulation")

    # Simulation parameters
    notional = st.number_input("Notional GBP exposure", value=1_000_000, step=50_000)
    days = st.slider("Simulation horizon (days)", min_value=30, max_value=180, value=90)
    n_sims = st.slider("Number of simulations", min_value=200, max_value=2000, value=500)

    # session storage
    if "agg_pnls" not in st.session_state:
        st.session_state.agg_pnls = None
        st.session_state.sims_dict = {}

    # Attempt to load precomputed results (non-blocking)
    if st.session_state.agg_pnls is None:
        try:
            summary_path = hf_download_to_path("sim_results/hedge_sim_summary.csv")
            scen_path = hf_download_to_path("sim_results/hedge_sim_scenarios_top200.csv")
            df_summary = pd.read_csv(summary_path, index_col=0)
            df_scenarios = pd.read_csv(scen_path, index_col=0)
            st.session_state.agg_pnls = df_summary
            st.session_state.sims_dict = df_scenarios
            st.info("✅ Loaded precomputed simulation results from Hugging Face.")
        except Exception:
            st.info("No precomputed results found on HF (or access denied). You can run a new simulation below.")

    # Button to run a fresh simulation
    run_sim = st.button("Run New Simulation")

    # Run simulation logic (inline; uses commodities DataFrame already loaded)
    if run_sim:
        if commodities.empty:
            st.error("Commodity data not available; cannot run simulation.")
        else:
            basket = {
                "wheat_spot": {"share": 0.6},
                "dairy_spot": {"share": 0.3},
                "oilseed_spot": {"share": 0.1},
            }
            sims_dict = {}
            agg_pnls = None

            try:
                for com_name, info in basket.items():
                    if com_name not in commodities.columns:
                        raise KeyError(f"{com_name} missing from commodity dataset.")
                    series = commodities[com_name]
                    sims = residual_bootstrap_sim(series, n_days=days, n_sims=n_sims)
                    sims_dict[com_name] = sims
                    basket[com_name]["last_price"] = series.dropna().iloc[-1]

                # aggregate pnl
                for com_name, info in basket.items():
                    pnl = compute_pnl_for_basket(sims_dict[com_name], info['last_price'], info['share'], notional)
                    agg_pnls = pnl if agg_pnls is None else agg_pnls + pnl

                # save to session
                st.session_state.agg_pnls = agg_pnls
                st.session_state.sims_dict = sims_dict
                st.success("✅ Simulation complete and saved in session.")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    # Display results (either precomputed DataFrame or freshly run np.array)
    if st.session_state.agg_pnls is not None:
        agg_pnls = st.session_state.agg_pnls
        sims_dict = st.session_state.sims_dict

        # If precomputed DataFrame (loaded from HF)
        if isinstance(agg_pnls, pd.DataFrame):
            df_summary = agg_pnls.copy()
            st.line_chart(df_summary[['p5', 'p50', 'p95']])
        else:
            # compute summary from array
            df_summary = pd.DataFrame({
                "mean": agg_pnls.mean(axis=0),
                "p5": np.percentile(agg_pnls, 5, axis=0),
                "p50": np.percentile(agg_pnls, 50, axis=0),
                "p95": np.percentile(agg_pnls, 95, axis=0),
            })
            st.line_chart(df_summary[['p5', 'p50', 'p95']])

        st.write("### Summary at key horizons")
        for d in [29, 59, min(89, df_summary.shape[0]-1)]:
            st.write(f"Day {d+1}: mean {df_summary['mean'].iloc[d]:,.0f}, p5 {df_summary['p5'].iloc[d]:,.0f}, p95 {df_summary['p95'].iloc[d]:,.0f}")

        # show top scenarios
        st.subheader("Top scenarios (preview)")
        if isinstance(sims_dict, pd.DataFrame):
            # precomputed scenarios file layout assumed: commodity column + day columns
            for com in sims_dict['commodity'].unique():
                st.write(f"**{com}**")
                st.dataframe(sims_dict[sims_dict['commodity'] == com].head(5).iloc[:, :10])
        else:
            # sims_dict is a dict of arrays
            for com, arr in sims_dict.items():
                st.write(f"**{com}**")
                preview = pd.DataFrame(arr[:5, :min(10, arr.shape[1])], columns=[f"Day {i+1}" for i in range(min(10, arr.shape[1]))])
                st.dataframe(preview)

# -----------------------------
# RAG + LLM Explainer (single-store S01)
# -----------------------------
elif view_choice == "RAG + LLM Explainer":
    st.subheader("💬 Ask Questions About Store S01")

    # Ensure local knowledge directory exists
    KNOW_DIR = "knowledge"
    os.makedirs(KNOW_DIR, exist_ok=True)

    # Try to fetch KB from Hugging Face (if available)
    from huggingface_hub import hf_hub_download
    kb_path = os.path.join(KNOW_DIR, "kb.csv")
    try:
        hf_path = hf_hub_download(
            repo_id=DATA_REPO_ID,
            repo_type=DATA_REPO_TYPE,
            filename="knowledge/kb.csv",
            token=st.secrets.get("HF_TOKEN", None)
        )
        if not os.path.exists(kb_path):
            os.replace(hf_path, kb_path)
        st.success("Knowledge base loaded from Hugging Face ✅")
    except Exception as e:
        st.info(f"knowledge/kb.csv not found on Hugging Face. A local KB will be built. ({e})")

    # Try to import helper module
    try:
        from rag_llm_explainer import build_knowledge_base, retrieve, call_openai_with_context

        # Ensure local KB exists
        if not os.path.exists(kb_path):
            with st.spinner("Building local knowledge base..."):
                build_knowledge_base()
                st.success("Knowledge base created locally ✅")

        # UI for user question
        query = st.text_input(
            "Enter your question about Store S01:",
            value="Why did store S01 see a drop in units last week?"
        )

        if "rag_answer" not in st.session_state:
            st.session_state.rag_answer = None
            st.session_state.rag_context = []

        if st.button("Get Answer") and query.strip():
            with st.spinner("Retrieving context and generating answer..."):
                context = retrieve(query)
                st.session_state.rag_context = context
                if context:
                    answer = call_openai_with_context(query, context)
                    st.session_state.rag_answer = answer
                else:
                    st.session_state.rag_answer = None

        # Display results
        if st.session_state.rag_answer:
            if st.session_state.rag_context:
                st.write("**Retrieved Context:**")
                for c in st.session_state.rag_context:
                    st.write(f"- {c}")
            st.write("**LLM Answer / Recommendation:**")
            st.success(st.session_state.rag_answer)
        elif st.session_state.rag_answer is None and st.session_state.rag_context == []:
            st.info("Enter a question and click **Get Answer** to see recommendations.")
        else:
            st.warning("No relevant context found in KB.")

    except ModuleNotFoundError:
        st.error("RAG helper module (`rag_llm_explainer.py`) not available. This feature is optional.")
    except Exception as e:
        st.error(f"Unexpected RAG error: {e}")

