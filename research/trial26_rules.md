# Trial #26: Risk-Adjusted Sector Momentum (Pre-Registered)
STATUS: FROZEN.

## Rules
- Core strategy: Exactly the same 11-sector rotation, Skip-Month 120d return, Sector-Level 200-SMA gate, equal weight (33.33%).
- Universe: 11 SPDR Sectors. Missing sectors are ineligible.

## Metrics & Evaluation
- **Primary Metric:** Calmar Ratio (CAGR / Max Drawdown).
- **Secondary Metric:** Sharpe Ratio.
- **Benchmark:** 1,000 runs of the clean, unlevered random baseline (randomly select 3 of 11 sectors, equal weight, no 200-SMA gate).
- **Pass Criteria:** The strategy's OOS Calmar Ratio must exceed the 95th percentile Calmar Ratio of the random baseline. (Decided before running the test).
