# Institutional Cross-Sectional Sector Momentum (Trial #24)
STATUS: FROZEN.

## 1. Universe & Data
- Assets: All 11 SPDR Sectors (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC).
- Data Handling: If a sector did not exist at a given date (e.g., XLC pre-2018), it is strictly ineligible. Do not zero-fill or backfill.

## 2. Monthly Evaluation (First Trading Day)
- Evaluate eligibility ONCE per month. Do not check intra-month.
- Crash Gate: Sector is eligible ONLY IF Close(t) > SMA200(t).

## 3. Momentum Ranking (Skip-Month)
- Metric: Skip-Month 120-Day Return. Formula: `(Price[t-20] / Price[t-120]) - 1`. (This avoids buying short-term blow-off tops).
- Select Top 3 eligible sectors.

## 4. Position Sizing
- Equal Weight (33.33% each).
- If fewer than 3 sectors are eligible, allocate equally to the eligible ones and hold the remainder in cash.
- Rebalance monthly. Apply 10bp cost per trade.
