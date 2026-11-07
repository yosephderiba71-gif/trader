# Multi-Asset Time-Series Momentum: Frozen Rules
STATUS: FROZEN.

## 1. Universe & Data
- Assets: SPY, TLT, GLD.
- Data: Daily bars (adjusted close).

## 2. Regime & Absolute Momentum Filter
- An asset is eligible ONLY IF: `Close(t) > SMA200(t)`.
- If ZERO assets pass: Allocate 100% of capital to cash (or BIL).

## 3. Relative Strength Ranking
- Among eligible assets, rank by 60-day return: `R(i,t) = (P(i,t) / P(i,t-60)) - 1`.
- Select Top K = 2 assets.
- Tie-breaker: If returns are identical, choose the asset with the lowest 60-day annualized volatility. If still tied, alphabetical.

## 4. Volatility Targeting & Position Sizing
- Metric: 60-day annualized realized volatility (std dev of log returns, multiplied by sqrt(252)).
- Raw weight: `w_raw(i) = (1/vol_i) / sum(1/vol_j)` for all j in K.
- Maximum weight cap: `w_max = 0.50`. If raw weight > 0.50, truncate to 0.50 and redistribute excess to the other asset. If only 1 asset passes, cap at 50% and hold 50% cash.

## 5. Execution
- Source of Truth: Query Alpaca `api.list_positions()` for current state. Do not use flat files.
- Rebalance Sequence: Sell assets not in target portfolio first, then buy new targets.
