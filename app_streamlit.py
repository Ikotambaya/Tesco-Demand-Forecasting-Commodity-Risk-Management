"""
Retail Demand & Commodity Risk Dashboard
📦 All data, models, and knowledge hosted on Hugging Face.
Repository: Uyane/tesco-project

Run with:
    streamlit run app_streamlit.py
"""

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
REPO_ID = "Uyane/tesco-project"   # dataset repository
REPO_TYPE = "dataset"             # ✅ tell Hugging Face it’s a dataset

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
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,              # ✅ dataset flag
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
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,              # ✅ dataset flag
            filename=filename,
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
store_choice = (
    st.sidebar.selectbox("Select Store", stores["store_id"].unique())
    if not stores.empty else None
)

view_choice = st.sidebar.radio(
    "Choose View",
    [
        "Store Insights",
        "Top SKUs",
        "Sales Insights",
        "Commodity Insights",
        "Forecasts",
        "Hedging Simulator",
        "RAG + LLM Explainer",
    ],
)

# -----------------------------
# STORE INSIGHTS
# -----------------------------
if view_choice == "Store Insights" and not agg.empty:
    st.subheader(f"📈 Store Demand Insights — {store_choice}")
    years = st.slider("Years to display", 1, 30, 15)
    end_date = agg["date"].max()
    start_date = end_date - pd.DateOffset(years=years)

    df = agg[(agg["store_id"] == store_choice) & (agg["date"] >= start_date)]
    if df.empty:
        st.warning("No data available for this selection.")
    else:
        total_units = df["units_sold"].sum()
        avg_price = df["price"].mean()
        promo_days = df["on_promo"].sum()
        stockouts = df["stockout"].sum()
        total_revenue = (df["units_sold"] * df["price"]).sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Units Sold", f"{total_units:,.0f}")
        c2.metric("Avg. Price", f"{avg_price:,.2f}")
        c3.metric("Promo Days", f"{promo_days:,}")
        c4.metric("Stockouts", f"{stockouts:,}")
        c5.metric("Total Revenue", f"${total_revenue:,.0f}")

        yearly = (
            df.groupby(df["date"].dt.year)["units_sold"]
            .sum()
            .reset_index(name="units_sold")
        )
        yearly["year"] = pd.to_datetime(yearly["date"].astype(str) + "-01-01")
        st.line_chart(yearly.set_index("year")["units_sold"])
        st.caption("Yearly total units sold")

# -----------------------------
# TOP SKUs
# -----------------------------
elif view_choice == "Top SKUs" and not sales.empty:
    st.subheader(f"🏆 Top SKUs — {store_choice}")
    df = sales[sales["store_id"] == store_choice]
    top_skus = (
        df.groupby("sku_id")["units_sold"].sum().sort_values(ascending=False).head(20)
    )
    st.bar_chart(top_skus)
    st.dataframe(skus[skus["sku_id"].isin(top_skus.index)])

# -----------------------------
# SALES INSIGHTS
# -----------------------------
elif view_choice == "Sales Insights" and not sales.empty:
    st.subheader(f"📊 Promotion Impact — {store_choice}")
    df = sales[sales["store_id"] == store_choice]

    categories = ["All"] + skus["category"].dropna().unique().tolist()
    cat_choice = st.selectbox("Filter by Category", categories)

    if cat_choice != "All":
        sku_ids = skus[skus["category"] == cat_choice]["sku_id"]
        df = df[df["sku_id"].isin(sku_ids)]

    promo_stats = df.groupby("on_promo")["units_sold"].mean().reset_index()
    promo_stats["promo_label"] = promo_stats["on_promo"].map({0: "No Promo", 1: "Promo"})

    chart = (
        alt.Chart(promo_stats)
        .mark_bar()
        .encode(
            x="promo_label:N",
            y="units_sold:Q",
            color="promo_label:N",
            tooltip=["promo_label", "units_sold"],
        )
        .properties(width=400, height=300, title="Avg Units Sold: Promo vs No Promo")
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# COMMODITY INSIGHTS
# -----------------------------
elif view_choice == "Commodity Insights" and not commodities.empty:
    st.subheader("🌾 Commodity Market Trends")
    com_choice = st.selectbox("Commodity", ["wheat_spot", "dairy_spot", "oilseed_spot"])
    years = st.slider("Years to display", 1, 30, 5)
    end_date = commodities["date"].max()
    start_date = end_date - pd.DateOffset(years=years)
    com_series = commodities.set_index("date")[com_choice]

    st.line_chart(com_series.loc[start_date:end_date])
    st.caption("Commodity spot price")

    vol = com_series.pct_change().rolling(90).std()
    st.line_chart(vol.loc[start_date:end_date])
    st.caption("Rolling 90-day volatility")

    st.subheader("📈 Correlation Matrix (3 years)")
    recent = commodities[commodities["date"] >= end_date - pd.DateOffset(years=3)]
    corr = recent[["wheat_spot", "dairy_spot", "oilseed_spot"]].pct_change().corr()
    st.dataframe(corr)

# -----------------------------
# FORECASTS
# -----------------------------
elif view_choice == "Forecasts":
    st.subheader("🔮 Machine Learning Forecasts")
    tab1, tab2 = st.tabs(["Store Demand Forecast", "Commodity Forecast"])

    with tab1:
        model = load_model("models/rf_store_nextday.joblib")
        if model and not agg.empty:
            df = agg[agg["store_id"] == store_choice].sort_values("date")
            df["dow"] = df["date"].dt.weekday
            for lag in [1, 7, 14]:
                df[f"lag_{lag}"] = df["units_sold"].shift(lag)
            df["ma_7"] = df["units_sold"].rolling(7, min_periods=1).mean()
            df = df.dropna()
            feats = [
                "lag_1", "lag_7", "lag_14", "ma_7",
                "on_promo", "stockout", "price", "avg_temp", "dow",
            ]
            preds = model.predict(df[feats].tail(30))
            st.line_chart(pd.Series(preds, index=df["date"].tail(30)))

    with tab2:
        model = load_model("models/rf_wheat_nextday.joblib")
        if model and not commodities.empty:
            df = commodities.copy().sort_values("date")
            df["lag_1"] = df["wheat_spot"].shift(1)
            df["lag_7"] = df["wheat_spot"].shift(7)
            df["ma_7"] = df["wheat_spot"].rolling(7, min_periods=1).mean()
            df = df.dropna()
            preds = model.predict(df[["lag_1", "lag_7", "ma_7"]].tail(30))
            st.line_chart(pd.Series(preds, index=df["date"].tail(30)))

# -----------------------------
# HEDGING SIMULATOR
# -----------------------------
elif view_choice == "Hedging Simulator":
    st.subheader("🛡️ Hedging Simulation")
    from hedging_simulator import load_commodities_safe, residual_bootstrap_sim, compute_pnl_for_basket

    notional = st.number_input("Notional GBP Exposure", value=1_000_000)
    days = st.slider("Simulation Horizon (days)", 30, 180, 90)
    n_sims = st.slider("Number of Simulations", 200, 2000, 500)

    if st.button("Run Simulation"):
        comm = load_commodities_safe(token=st.secrets["HF_TOKEN"])
        basket = {
            "wheat_spot": {"share": 0.6},
            "dairy_spot": {"share": 0.3},
            "oilseed_spot": {"share": 0.1},
        }
        sims_dict, agg_pnls = {}, None
        for col in basket:
            sims = residual_bootstrap_sim(comm[col], n_days=days, n_sims=n_sims)
            sims_dict[col] = sims
            basket[col]["last_price"] = comm[col].iloc[-1]
        for col, info in basket.items():
            pnl = compute_pnl_for_basket(sims_dict[col], info, notional)
            agg_pnls = pnl if agg_pnls is None else agg_pnls + pnl

        df_summary = pd.DataFrame({
            "p5": np.percentile(agg_pnls, 5, axis=0),
            "p50": np.percentile(agg_pnls, 50, axis=0),
            "p95": np.percentile(agg_pnls, 95, axis=0),
        })
        st.line_chart(df_summary[["p5", "p50", "p95"]])
        st.success("Simulation complete ✅")

# -----------------------------
# RAG + LLM EXPLAINER
# -----------------------------
elif view_choice == "RAG + LLM Explainer":
    st.subheader("💬 Ask Questions About Store S01")
    from rag_llm_explainer import build_knowledge_base, retrieve, call_openai_with_context

    kb_path = hf_hub_download(
        repo_id=REPO_ID, filename="knowledge/kb.csv", token=st.secrets["HF_TOKEN"]
    )

    query = st.text_input(
        "Enter your question:",
        value="Why did store S01 see a drop in sales last week?",
    )

    if st.button("Get Answer"):
        context = retrieve(query)
        if context:
            answer = call_openai_with_context(query, context)
            st.write("**Retrieved Context:**")
            for c in context:
                st.write("-", c)
            st.write("**LLM Answer:**")
            st.success(answer)
        else:
            st.warning("No relevant context found.")

