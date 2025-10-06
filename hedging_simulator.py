# hedging_simulator.py
"""
Robust Monte-Carlo Hedging Simulator for Commodity Exposure
- Supports dynamic baskets and safe loading of historical data
- Residual bootstrapped AR(1) simulations
- Computes P&L per commodity and aggregates
- Saves scenario summaries and top scenarios
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("data")
OUT_DIR = Path("sim_results")
OUT_DIR.mkdir(exist_ok=True)

# -------------------------------
# Load Commodities Safely
# -------------------------------
def load_commodities_safe() -> pd.DataFrame:
    """
    Loads and validates commodities CSV. Returns DataFrame indexed by date.
    """
    path = DATA_DIR / "commodities.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing commodity CSV: {path}")
    comm = pd.read_csv(path, parse_dates=['date'])
    required_cols = ['wheat_spot','dairy_spot','oilseed_spot']
    missing = [c for c in required_cols if c not in comm.columns]
    if missing:
        raise ValueError(f"Missing columns in commodities CSV: {missing}")
    comm = comm.sort_values('date').set_index('date')
    return comm

# -------------------------------
# Residual Bootstrapped AR(1)
# -------------------------------
def residual_bootstrap_sim(series: pd.Series, n_days: int = 90, n_sims: int = 2000) -> np.ndarray:
    """
    Simulates future price paths using AR(1) with bootstrapped residuals.
    """
    s = series.dropna()
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
        path = [last]
        for d in range(n_days):
            e = np.random.choice(resid.values)
            nxt = path[-1] + mu + e
            path.append(nxt)
        sims[i, :] = path[1:]
    return sims

# -------------------------------
# Compute P&L per commodity
# -------------------------------
def compute_pnl_for_basket(sims: np.ndarray, weight: Dict[str, Any], notional: float) -> np.ndarray:
    """
    Computes P&L for a single commodity basket.
    """
    last = weight.get('last_price')
    share = weight.get('share')
    if last is None or share is None or not (0 <= share <= 1):
        raise ValueError("weight dict must contain 'last_price' and valid 'share'")
    pct_changes = (sims - last) / last
    pnl = pct_changes * (notional * share)
    return pnl

# -------------------------------
# Run Simulation
# -------------------------------
def run_hedging_sim(
    basket: Dict[str, Dict[str, Any]] = None,
    notional: float = 1_000_000,
    n_days: int = 90,
    n_sims: int = 2000,
    random_seed: int = None
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Runs hedging simulation for given basket and parameters.
    Returns:
        - Aggregate P&L (n_sims x n_days)
        - Summary DataFrame (mean, p5, p50, p95 per key day)
        - Individual commodity simulations
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if basket is None:
        basket = {
            'wheat_spot': {'share':0.6},
            'dairy_spot': {'share':0.3},
            'oilseed_spot': {'share':0.1}
        }

    comm = load_commodities_safe()
    sims_dict = {}

    # Run simulations for each commodity
    for col, info in basket.items():
        if col not in comm.columns:
            logging.warning(f"{col} not found in commodity data, skipping")
            continue
        sims = residual_bootstrap_sim(comm[col], n_days=n_days, n_sims=n_sims)
        sims_dict[col] = sims
        basket[col]['last_price'] = comm[col].iloc[-1]

    # Aggregate P&L
    agg_pnls = np.zeros((n_sims, n_days))
    for col, info in basket.items():
        pnl = compute_pnl_for_basket(sims_dict[col], info, notional)
        agg_pnls += pnl

    # Summarize key days
    summary_days = [29, 59, min(89, n_days-1)]  # 30d, 60d, 90d
    summary = {}
    for day_idx in summary_days:
        arr = agg_pnls[:, day_idx]
        summary[f"Day {day_idx+1}"] = {
            "mean": float(np.mean(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95))
        }

    summary_df = pd.DataFrame(summary).T

    # Save outputs
    try:
        summary_df.to_csv(OUT_DIR / "hedge_sim_summary.csv")
        pd.DataFrame(agg_pnls[:200, :n_days]).to_csv(OUT_DIR / "hedge_sim_scenarios_top200.csv", index=False)
    except Exception as e:
        logging.error(f"Error saving simulation outputs: {e}")

    logging.info(f"Hedging simulation complete. Summary:\n{summary_df}")
    return agg_pnls, summary_df, sims_dict

# -------------------------------
# Standalone run
# -------------------------------
if __name__ == "__main__":
    run_hedging_sim()
