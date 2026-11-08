# Institutional Sector Momentum with Volatility Targeting (Trial #25)
STATUS: FROZEN.

## 1. Core Rules (Unchanged from Trial #24)
- Universe: 11 SPDR Sectors.
- Monthly Rebalance, Skip-Month 120-day return, Top 3 sectors.
- Sector-Level 200-SMA gate (drop to cash if below SMA).

## 2. Volatility Targeting (Tobin's Separation Theorem)
- Target Annualized Volatility: 15%.
- Basket Volatility Calculation: Do NOT average individual sector vols. Calculate equal-weighted (1/3 each) daily returns over the past 60 days, then annualize the standard deviation (multiply by sqrt(252)).
- Scalar = (0.15 / basket_volatility).
- Bound the Scalar: Minimum 0.50x, Maximum 1.35x.

## 3. Costs & Leverage
- Base Trade Cost: 10bp.
- Margin Cost: If Scalar > 1.0, apply a 5% annual borrowing rate on the margin debit. Deduct this monthly.
