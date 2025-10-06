# train_baselines.py
"""
Train baseline RandomForest models:
 - store next-day total units (Tesco Rubery, S01)
 - commodity (wheat) next-day spot price
Saves models to ./models/
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

STORE_ID = "S01"   # Tesco Superstore, Rubery

def train_store_model():
    df = pd.read_csv(os.path.join(DATA_DIR, "daily_store_agg.csv"), parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ restrict to Rubery store only
    df = df[df['store_id'] == STORE_ID].copy()

    # sort and create features
    df = df.sort_values(['date'])
    df['dow'] = df['date'].dt.weekday

    # lag features
    for lag in [1, 7, 14]:
        df[f'units_lag_{lag}'] = df['units_sold'].shift(lag)

    # rolling average
    df['ma_7'] = df['units_sold'].rolling(7, min_periods=1).mean()

    # drop rows without lag data
    df = df.dropna().reset_index(drop=True)

    # target: next-day units
    df['units_t1'] = df['units_sold'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    feats = [
        'units_lag_1', 'units_lag_7', 'units_lag_14',
        'ma_7', 'on_promo', 'stockout', 'price', 'avg_temp', 'dow'
    ]
    X = df[feats]
    y = df['units_t1']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, shuffle=False
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_store_nextday.joblib"))
    print(f"[Store model] MAE: {mae:.2f}, RMSE: {rmse:.2f}")


def train_commodity_model():
    comm = pd.read_csv(os.path.join(DATA_DIR, "commodities.csv"), parse_dates=['date'])
    comm['date'] = pd.to_datetime(comm['date'], errors='coerce')
    comm = comm.sort_values('date')

    # features
    comm['wheat_lag_1'] = comm['wheat_spot'].shift(1)
    comm['wheat_lag_7'] = comm['wheat_spot'].shift(7)
    comm['wheat_ma_7'] = comm['wheat_spot'].rolling(7, min_periods=1).mean()

    # target
    comm = comm.dropna().reset_index(drop=True)
    comm['wheat_t1'] = comm['wheat_spot'].shift(-1)
    comm = comm.dropna().reset_index(drop=True)

    feats = ['wheat_lag_1', 'wheat_lag_7', 'wheat_ma_7']
    X = comm[feats]
    y = comm['wheat_t1']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, shuffle=False
    )

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_wheat_nextday.joblib"))
    print(f"[Wheat model] MAE: {mae:.2f}, RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train_store_model()
    train_commodity_model()
    print("✅ Models saved to ./models/")
