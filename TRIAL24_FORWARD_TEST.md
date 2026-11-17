# Trial #24 Forward Test

## Status

Experimental paper-trading forward test. No live capital is used.

## Account isolation

Trial #24 runs on a fully isolated, dedicated Alpaca paper account.

- The runner reads only `TRIAL24_ALPACA_API_KEY` and `TRIAL24_ALPACA_SECRET_KEY`.
- It never reads or falls back to the main Discipline Bot credentials
  (`ALPACA_API_KEY` / `ALPACA_API_SECRET`).
- The GitHub Actions workflow injects the Trial #24 values from repository
  secrets. Secret values are never committed to the repository.
- A hard `paper=True` assertion runs before the trading client is constructed.
- The dedicated account must not be shared with any other bot or manual trading.

## Strategy rules

Trial #24 is the frozen unlevered cross-sectional sector momentum strategy:

- Universe: the 11 SPDR sector ETFs plus SPY as the benchmark.
- Signal: skip-month 120-day total return.
- Eligibility: sector price must be above its 200-day simple moving average.
- Selection: top 3 eligible sectors.
- Weighting: equal 1/3 each; unallocated capital remains in cash.
- Rebalance: first trading day of each month.
- Exposure: maximum 100% gross; no margin or leverage.
- Execution: market orders through the dedicated paper account.

Rules must remain unchanged during the forward test. Any proposed change
restarts the test period and must be documented before implementation.

## Benchmark

Performance is compared with two references:

1. **SPY buy-and-hold** — the primary opportunity-cost benchmark.
2. **SPY + Gate** — SPY exposure only while SPY closes above its 200-day
   simple moving average, otherwise cash. This isolates whether the Trial #24
   sector selection adds value beyond a simple index trend filter.

Report the strategy against both benchmarks over the same dates.

## Timeline

- Forward-test horizon: **3-5 years (36-60 months)** from the first live paper
  rebalance.
- Rebalancing cadence: monthly, first trading day.
- No early promotion based on a short window; interim results are observational.
- The start date is the first trading day after the dedicated account and
  GitHub secrets are configured.

## Metrics to track

- Total and annualized return
- Volatility
- Sharpe ratio
- Maximum drawdown and Calmar ratio
- Monthly hit rate
- Turnover and estimated transaction costs
- Excess return versus SPY buy-and-hold
- Excess return versus SPY + Gate
- Tracking of realized weights versus target weights

## Operational guardrails

- The monthly workflow installs dependencies and runs the full test suite before
  any rebalance attempt.
- Scheduled runs use `--execute`; manual dispatches default to `--dry-run`.
- The runner enforces the first-trading-day guard before submitting orders.
- A failure to fetch prices, authenticate, or reconcile positions aborts the run
  without discretionary intervention.
- No leverage, no margin, and no manual overrides are permitted during the test.
