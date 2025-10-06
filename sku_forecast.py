# sku_forecast.py
"""
SKU-level forecasting:
 - Train per-SKU time-series models for Tesco Superstore, Rubery (S01).
 - Saves models and a metrics CSV to ./models/sku_models/
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = "data"
MODEL_SUBDIR = "models/sku_models"
os.makedirs(MODEL_SUBDIR, exist_ok=True)

STORE_ID = "S01"   # fixed to Tesco Superstore, Rubery, Birmingham

def prepare_sku_series(sales_df, store_id, sku_id):
    df = sales_df[(sales_df['store_id']==store_id)&(sales_df['sku_id']==sku_id)].copy()
    df = df.sort_values('date')
    df['date'] = pd.to_datetime(df['date'])
    
    # aggregate if duplicates
    df = df.groupby('date').agg({
        'units_sold':'sum',
        'on_promo':'max',
        'price':'mean',
        'stockout':'max',
        'temp':'mean'
    }).reset_index()

    # create lag features
    for lag in [1,7,14]:
        df[f'lag_{lag}'] = df['units_sold'].shift(lag)

    # rolling mean
    df['ma_7'] = df['units_sold'].rolling(7, min_periods=1).mean()

    # day-of-week
    df['dow'] = df['date'].dt.weekday

    # drop rows without enough history
    df = df.dropna().reset_index(drop=True)

    # target variable = next-day demand
    df['units_t1'] = df['units_sold'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    
    return df

def train_for_sku(df):
    feats = [c for c in df.columns if c.startswith('lag_')] + [
        'ma_7','on_promo','stockout','price','temp','dow'
    ]
    X = df[feats]; y = df['units_t1']
    if len(X) < 60:  # skip SKUs with too little history
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return model, mae

if __name__ == "__main__":
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales_data.csv"), parse_dates=['date'])
    top_n = 10  # train on top 10 SKUs by sales
    top_skus = (
        sales[sales['store_id']==STORE_ID]
        .groupby('sku_id')['units_sold']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    metrics = []
    for sku in top_skus:
        df = prepare_sku_series(sales, STORE_ID, sku)
        model, mae = train_for_sku(df)
        if model is not None:
            model_path = os.path.join(MODEL_SUBDIR, f"sku_{STORE_ID}_{sku}.joblib")
            joblib.dump(model, model_path)
            metrics.append({"store_id":STORE_ID,"sku_id":sku,"mae":mae,"n_rows":len(df)})
            print(f"✅ Trained {sku} | MAE: {mae:.2f} | rows:{len(df)}")
        else:
            metrics.append({"store_id":STORE_ID,"sku_id":sku,"mae":None,"n_rows":len(df)})
            print(f"⚠️ Skipped {sku} (not enough rows: {len(df)})")

    pd.DataFrame(metrics).to_csv(os.path.join(MODEL_SUBDIR, "sku_metrics.csv"), index=False)
    print("📦 SKU models and metrics saved to", MODEL_SUBDIR)
