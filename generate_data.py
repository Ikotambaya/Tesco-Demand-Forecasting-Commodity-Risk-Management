# generate_data.py
"""
Generate synthetic Tesco retail + commodity datasets for one store
(Tesco Superstore, Rubery, Birmingham).
Spans 30 years: 1995-01-01 to 2025-09-30.
Saves CSVs to ./data/
"""
import os
import numpy as np
import pandas as pd

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

def make_store():
    """Single Tesco Superstore: Rubery, Birmingham"""
    store = pd.DataFrame([{
        "store_id": "S01",
        "store_name": "Tesco Superstore, Rubery, Birmingham",
        "region": "West Midlands",
        "latitude": 52.3936,
        "longitude": -1.9943
    }])
    store.to_csv(os.path.join(OUT_DIR, "stores.csv"), index=False)
    return store

def make_skus(n_skus=200):
    categories = ["Bakery", "Dairy", "Beverages", "Produce", "Frozen", "Pantry"]
    skus = pd.DataFrame({
        "sku_id": [f"SKU{str(i).zfill(4)}" for i in range(1, n_skus+1)],
        "category": np.random.choice(categories, n_skus, 
                                     p=[0.12,0.18,0.18,0.20,0.12,0.20]),
        "base_price": np.round(np.random.uniform(0.5, 10.0, n_skus), 2)
    })
    skus.to_csv(os.path.join(OUT_DIR, "skus.csv"), index=False)
    return skus

def generate_sales(store, skus, start_date="1995-01-01", end_date="2025-09-30"):
    dates = pd.date_range(start_date, end_date, freq="D")

    rows = []
    for _, sku in skus.iterrows():
        cat_factor = {"Bakery":1.2,"Dairy":1.1,"Beverages":1.0,
                      "Produce":1.15,"Frozen":0.9,"Pantry":1.0}[sku['category']]
        base_daily = np.random.poisson(3) * cat_factor + (sku['base_price'] < 2.0) * 1.5

        for day in dates:
            dow = day.weekday()
            is_weekend = int(dow >= 5)
            month = day.month

            # seasonal boosts
            holiday_boost = 1.0
            if (month == 12 and day.day >= 20) or (month == 1 and day.day <=7):
                holiday_boost = 1.6  # Christmas / New Year
            if month == 4 and 1 <= day.day <= 14:
                holiday_boost = max(holiday_boost, 1.2)  # Easter

            promo = np.random.rand() < 0.03
            promo_multiplier = 1.8 if promo else 1.0
            stockout = np.random.rand() < 0.005
            price = sku['base_price'] * (1 + 0.02 * np.random.randn())
            price_effect = max(0.5, 1.0 - 0.07*(price - sku['base_price']))

            temp = 8 + 10*np.sin(2*np.pi*(day.timetuple().tm_yday)/365) + np.random.randn()*3

            mean_demand = max(0.0, base_daily * (1 + 0.15*is_weekend) * 
                              holiday_boost * promo_multiplier * price_effect)
            demand = np.random.poisson(max(0.1, mean_demand))

            if np.random.rand() < 0.001:  # rare demand spikes
                demand = int(demand * np.random.uniform(3,8))

            sold = 0 if stockout else int(demand)

            rows.append({
                "date": day,
                "store_id": "S01",
                "sku_id": sku['sku_id'],
                "category": sku['category'],
                "price": round(price,2),
                "on_promo": int(promo),
                "stockout": int(stockout),
                "units_sold": sold,
                "is_weekend": is_weekend,
                "holiday_boost": holiday_boost,
                "temp": round(temp,1)
            })

    sales = pd.DataFrame(rows)
    sales.to_csv(os.path.join(OUT_DIR, "sales_data.csv"), index=False)
    return sales

def generate_weather(store, start_date="1995-01-01", end_date="2025-09-30"):
    dates = pd.date_range(start_date, end_date, freq="D")
    weather_rows = []
    lat = store['latitude'].iloc[0]

    for day in dates:
        doy = day.timetuple().tm_yday
        temp = 8 + 10*np.sin(2*np.pi*doy/365) + np.random.randn()*3 + (lat-52.0)*0.1
        precip = max(0, np.random.exponential(0.5) - 0.2*(np.cos(2*np.pi*doy/365)))
        weather_rows.append({"date": day, "store_id": "S01", 
                             "temp": round(temp,1), "precip_mm": round(precip,2)})

    weather = pd.DataFrame(weather_rows)
    weather.to_csv(os.path.join(OUT_DIR, "weather.csv"), index=False)
    return weather

def generate_commodities(start_date="1995-01-01", end_date="2025-09-30"):
    dates = pd.date_range(start_date, end_date, freq="D")
    rows = []
    for day in dates:
        t = (day - dates[0]).days
        wheat = 180 + 0.01*t + 10*np.sin(2*np.pi*(day.timetuple().tm_yday)/365) + np.random.randn()*4
        dairy = 150 + 0.008*t + 8*np.sin(2*np.pi*(day.timetuple().tm_yday)/365 + 0.5) + np.random.randn()*3
        oilseed = 400 + 0.02*t + 20*np.sin(2*np.pi*(day.timetuple().tm_yday)/365 + 1.0) + np.random.randn()*6
        rows.append({"date": day, "wheat_spot": round(wheat,2), 
                     "dairy_spot": round(dairy,2), "oilseed_spot": round(oilseed,2)})
    comm = pd.DataFrame(rows)
    comm.to_csv(os.path.join(OUT_DIR, "commodities.csv"), index=False)
    return comm

def make_aggregates(sales):
    agg = sales.groupby(['date','store_id']).agg({
        'units_sold': 'sum',
        'on_promo': 'sum',
        'stockout': 'sum',
        'price': 'mean',
        'temp': 'mean'
    }).reset_index()
    agg.rename(columns={'temp':'avg_temp'}, inplace=True)
    agg.to_csv(os.path.join(OUT_DIR, "daily_store_agg.csv"), index=False)
    return agg

if __name__ == "__main__":
    store = make_store()
    skus = make_skus(200)
    sales = generate_sales(store, skus)
    weather = generate_weather(store)
    comm = generate_commodities()
    agg = make_aggregates(sales)
    print("Data written to:", OUT_DIR)
    print("Store:", store.shape)
    print("SKUs:", skus.shape)
    print("Sales:", sales.shape)
    print("Weather:", weather.shape)
    print("Commodities:", comm.shape)
    print("Aggregated:", agg.shape)
