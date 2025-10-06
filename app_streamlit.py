# app_streamlit.py
"""
Retail Demand & Commodity Risk Dashboard
- Store demand insights (2010–2025 data)
- SKU performance & promotion lift
- Commodity trends, volatility & correlation
- ML-based forecasts (store + commodity)
- Hedging simulator (Monte Carlo)
- RAG + LLM explainer
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download
import textwrap

# -----------------
# Directories
# -----------------
DATA_DIR = "data"
KNOW_DIR = "knowledge"

# -----------------
# Title
# -----------------
st.set_page_config(page_title="Retail & Commodity Risk Demo", layout="wide")
st.title("📊 Retail Demand & Commodity Risk Dashboard")

# -----------------
# Load Data
# -----------------
@st.cache_data
def load_csv(name, parse_dates=None):
    return pd.read_csv(os.path.join(DATA_DIR, name), parse_dates=parse_dates)

stores = load_csv("stores.csv")
skus = load_csv("skus.csv")
sales = load_csv("sales_data.csv", parse_dates=['date'])
agg = load_csv("daily_store_agg.csv", parse_dates=['date'])
commodities = load_csv("commodities.csv", parse_dates=['date'])

# -----------------
# Sidebar controls
# -----------------
st.sidebar.header("🔎 Controls")
store_choice = st.sidebar.selectbox("Select Store", stores['store_id'].tolist())
view_choice = st.sidebar.radio(
    "Choose View",
    ["Store Insights", "Top SKUs", "Sales Insights", "Commodity Insights", "Forecasts", "Hedging Simulator", "RAG + LLM Explainer"]
)

# -----------------
# Store Insights
# -----------------
if view_choice == "Store Insights":
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
        promo_days = df['on_promo'].sum()
        stockouts = df['stockout'].sum()
        total_revenue = (df['units_sold'] * df['price']).sum()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Units Sold", f"{total_units:,.0f}")
        col2.metric("Avg. Price", f"{avg_price:,.2f}")
        col3.metric("Promo Days", f"{promo_days:,}")
        col4.metric("Stockouts", f"{stockouts:,}")
        col5.metric("Total Revenue", f"${total_revenue:,.0f}")

        yearly = (
            df.groupby(df['date'].dt.year)['units_sold']
              .sum()
              .reset_index(name="units_sold")
              .rename(columns={"date": "year"})
        )
        yearly['year'] = pd.to_datetime(yearly['year'].astype(str) + "-01-01")

        st.line_chart(yearly.set_index('year')['units_sold'])
        st.caption("Yearly total units sold (1995–2025)")

        st.write("Recent daily data:")
        st.dataframe(df.sort_values('date', ascending=False).head(15))

# -----------------
# Top SKUs
# -----------------
elif view_choice == "Top SKUs":
    st.subheader(f"🏆 Top SKUs by volume — {store_choice}")
    df = sales[sales['store_id']==store_choice]
    top_skus = df.groupby('sku_id')['units_sold'].sum().sort_values(ascending=False).head(20)
    st.bar_chart(top_skus)
    st.write("SKU details", skus[skus['sku_id'].isin(top_skus.index)])

# -----------------
# Sales Insights (promotion analysis)
# -----------------
elif view_choice == "Sales Insights":
    st.subheader(f"📊 Promotion Impact — {store_choice}")
    df_store = sales[sales['store_id']==store_choice]
    categories = ["All"] + skus['category'].unique().tolist()
    cat_choice = st.selectbox("Filter by Category", categories)
    if cat_choice != "All":
        sku_ids = skus[skus['category'] == cat_choice]['sku_id']
        df_filtered = df_store[df_store['sku_id'].isin(sku_ids)]
    else:
        df_filtered = df_store.copy()

    promo_stats = df_filtered.groupby('on_promo')['units_sold'].mean().reset_index()
    promo_stats['promo_label'] = promo_stats['on_promo'].map({0:'No Promo',1:'Promo'})
    st.write("Average units sold:")

    import altair as alt
    chart = alt.Chart(promo_stats).mark_bar().encode(
        x='promo_label:N',
        y='units_sold:Q',
        color='promo_label:N',
        tooltip=['promo_label','units_sold']
    ).properties(width=400,height=300,title=f"Average Units Sold: Promo vs No Promo ({cat_choice} category)")
    st.altair_chart(chart, use_container_width=True)

    try:
        no_promo = promo_stats.loc[promo_stats['on_promo']==0, 'units_sold'].values[0]
        promo = promo_stats.loc[promo_stats['on_promo']==1, 'units_sold'].values[0]
        lift_pct = ((promo - no_promo) / no_promo) * 100
    except IndexError:
        lift_pct = 0
    st.write(f"**Promotion Lift:** {lift_pct:.1f}%")

    gauge = alt.Chart(pd.DataFrame({'value':[lift_pct]})).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field='value', type='quantitative', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(field='value', type='quantitative', scale=alt.Scale(scheme='greens')),
        tooltip=['value']
    ).properties(width=200,height=200,title="Promo Lift %")
    st.altair_chart(gauge)
    st.write("Recent sales breakdown")
    st.dataframe(df_filtered.sort_values('date', ascending=False).head(15))

# -----------------
# Commodity Insights
# -----------------
elif view_choice == "Commodity Insights":
    st.subheader("🌾 Commodity Market Trends")
    com_choice = st.selectbox("Commodity", ['wheat_spot','dairy_spot','oilseed_spot'])
    years = st.slider("Years to display", 1, 30, 5)
    end_date = commodities['date'].max()
    start_date = end_date - pd.DateOffset(years=years)

    com_series = commodities.set_index('date')[com_choice]
    st.line_chart(com_series.loc[start_date:end_date])
    st.write("Recent values", com_series.tail(10))

    vol = com_series.pct_change().rolling(90).std()
    st.line_chart(vol.loc[start_date:end_date])
    st.caption("Rolling 90-day volatility")

    st.subheader("Commodity Correlation Matrix (last 3 years)")
    recent = commodities[commodities['date']>=end_date - pd.DateOffset(years=3)]
    corr = recent[['wheat_spot','dairy_spot','oilseed_spot']].pct_change().corr()
    st.dataframe(corr)

# -----------------
# Forecasts
# -----------------
elif view_choice == "Forecasts":
    st.subheader("🔮 ML Forecasts")
    tab1, tab2 = st.tabs(["Store Demand Forecast", "Commodity Forecast"])

    # Store Forecast
    with tab1:
        try:
            store_model_path = hf_hub_download(repo_id="Uyane/tesco-model", filename="models/rf_store_nextday.joblib")
            model = joblib.load(store_model_path)
            df_store = agg[agg['store_id']==store_choice].copy()
            df_store = df_store.sort_values('date')
            df_store['dow'] = df_store['date'].dt.weekday

            for lag in [1,7,14]:
                df_store[f'units_lag_{lag}'] = df_store['units_sold'].shift(lag)
            df_store['ma_7'] = df_store['units_sold'].rolling(7, min_periods=1).mean()
            df_store = df_store.dropna()

            feats = ['units_lag_1','units_lag_7','units_lag_14','ma_7','on_promo','stockout','price','avg_temp','dow']
            X_store = df_store[feats]
            y_pred = model.predict(X_store.tail(30))
            st.line_chart(pd.Series(y_pred, index=df_store['date'].tail(30), name="Forecasted Units"))

        except Exception as e:
            st.error(f"Store forecast model not available: {e}")

    # Commodity Forecast
    with tab2:
        try:
            comm_model_path = hf_hub_download(repo_id="Uyane/tesco-model", filename="models/rf_wheat_nextday.joblib")
            model = joblib.load(comm_model_path)
            df_comm = commodities.copy().sort_values('date')

            df_comm['wheat_lag_1'] = df_comm['wheat_spot'].shift(1)
            df_comm['wheat_lag_7'] = df_comm['wheat_spot'].shift(7)
            df_comm['wheat_ma_7'] = df_comm['wheat_spot'].rolling(7, min_periods=1).mean()
            df_comm = df_comm.dropna()

            feats = ['wheat_lag_1','wheat_lag_7','wheat_ma_7']
            X_comm = df_comm[feats]
            y_pred = model.predict(X_comm.tail(30))
            st.line_chart(pd.Series(y_pred, index=df_comm['date'].tail(30), name="Wheat Forecast"))

        except Exception as e:
            st.error(f"Commodity forecast model not available: {e}")

# -----------------
# Hedging Simulator
# -----------------
elif view_choice == "Hedging Simulator":
    st.subheader("🛡️ Hedging Simulation")
    notional = st.number_input("Notional GBP exposure", value=1_000_000, step=50_000)
    days = st.slider("Simulation horizon (days)", min_value=30, max_value=180, value=90)
    n_sims = st.slider("Number of simulations", min_value=200, max_value=2000, value=500)

    if st.button("Run Simulation"):
        from hedging_simulator import load_commodities_safe, residual_bootstrap_sim, compute_pnl_for_basket
        comm = load_commodities_safe()
        basket = {'wheat_spot': {'share':0.6}, 'dairy_spot': {'share':0.3}, 'oilseed_spot': {'share':0.1}}
        sims_dict, agg_pnls = {}, None
        for col in basket:
            sims = residual_bootstrap_sim(comm[col], n_days=days, n_sims=n_sims)
            sims_dict[col] = sims
            basket[col]['last_price'] = comm[col].iloc[-1]
        for col, info in basket.items():
            pnl = compute_pnl_for_basket(sims_dict[col], info, notional)
            agg_pnls = pnl if agg_pnls is None else agg_pnls + pnl

        df_summary = pd.DataFrame({
            "mean": agg_pnls.mean(axis=0),
            "p5": np.percentile(agg_pnls,5,axis=0),
            "p50": np.percentile(agg_pnls,50,axis=0),
            "p95": np.percentile(agg_pnls,95,axis=0),
        })
        st.line_chart(df_summary[['p5','p50','p95']])
        st.write("Summary at key horizons:")
        for d in [29,59,min(89,days-1)]:
            st.write(f"Day {d+1}: mean {df_summary['mean'].iloc[d]:,.0f}, "
                     f"p5 {df_summary['p5'].iloc[d]:,.0f}, "
                     f"p95 {df_summary['p95'].iloc[d]:,.0f}")
        st.success("Simulation complete ✅")

# -----------------
# RAG + LLM Explainer (single-store S01)
# -----------------
elif view_choice == "RAG + LLM Explainer":
    st.subheader("💬 Ask Questions About Store S01")
    from rag_llm_explainer import build_knowledge_base, retrieve, call_openai_with_context

    kb_path = os.path.join(KNOW_DIR, "kb.csv")
    if not os.path.exists(kb_path):
        build_knowledge_base()

    query = st.text_input(
        "Enter your question about Store S01:",
        value="Why did store S01 see a drop in units last week?"
    )

    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = None
        st.session_state.rag_context = []

    if st.button("Get Answer") and query.strip():
        context = retrieve(query)
        st.session_state.rag_context = context
        if context:
            st.session_state.rag_answer = call_openai_with_context(query, context)
        else:
            st.session_state.rag_answer = None

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
