# Trial #27: Turn-of-the-Month (TOTM) Effect (Pre-Registered)
STATUS: FROZEN.

## Rules
- Asset: SPY (Risky), BIL (Cash). Unlevered only.
- Entry: Buy SPY at the close on the last trading day of Month T (Day -1).
- Holding: Hold through trading days +1, +2, and +3 of Month T+1.
- Exit: Sell SPY at the close on trading day +3 and buy BIL.
- Execution: Standard market order at 15:50 EST.

## Metrics & Evaluation
- Primary Metric: Calmar Ratio and Sharpe Ratio.
- Benchmark: Buy-and-Hold SPY.
- Random Baseline: Do NOT sample from the entire dataset. For each month m, randomly select a starting day index k to form a random 4-day holding window *within that specific month*. Compound the 4-day return, add BIL's risk-free return for the remaining days. Generate 1,000 trajectories and find the 95th percentile.
