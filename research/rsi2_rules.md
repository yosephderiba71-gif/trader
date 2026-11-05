# RSI(2) Mean Reversion on SPY and QQQ: Frozen Rules

Status: FROZEN. Written before any code or data was run.
Any change after the commit = a new trial, logged as such.

## Data
- SPY and QQQ daily OHLC, dividend/split adjusted (yfinance auto_adjust=True).
- Start: 2000-01-01 (QQQ from its first available date). End: latest complete day.
- Chronological split by trading days: first 70% in-sample (sanity only),
  last 30% out-of-sample (OOS). The split date is computed once and recorded.
- No parameter is tuned on in-sample or OOS data.

## Indicators (computed on adjusted close)
- RSI(2) with Wilder smoothing.
- SMA(200) and SMA(5).
- No signal before 200 bars of history exist.

## Rules (long only, one position per instrument, no leverage, 100% notional)
- Entry signal at close of day t: close > SMA(200) AND RSI(2) < 10, no open position.
- Fill: open of day t+1.
- Exit signal at close of day t: close > SMA(5), OR 10 trading days held
  (whichever comes first).
- Fill: open of day t+1.
- No stop loss. The 10-day time stop bounds the holding period.
- Each instrument is tested separately, then pooled.

## Costs
- Round-trip cost 5bp (base case) and 10bp (stress case), subtracted from each trade return.
- Trade return = exit_open / entry_open - 1 - cost.

## Metrics (OOS, per instrument and pooled)
- Trades, win rate, average return after costs, profit factor,
  max drawdown, time in market, CAGR.
- Buy-and-hold CAGR and max drawdown over the same OOS window.
- Random baseline: same number of trades and the same holding durations,
  random entry dates, 1,000 runs. Report the percentile of the real
  average return.

## Pass criteria (ALL must hold at 10bp, OOS)
1. Pooled average trade return after costs > 0.
2. Pooled trades >= 100. If fewer, the result is "inconclusive", not a pass.
3. Pooled profit factor >= 1.2.
4. Real average return at or above the 95th percentile of the random baseline.
5. Positive average return on BOTH SPY and QQQ separately.
6. Max drawdown lower than buy-and-hold's over the OOS window.

## If it fails
- Verdict: rejected. Log in REPORT.md. No retuning of thresholds, SMA length,
  or exit.
- Any variant (e.g. RSI < 5, SMA(100)) counts as a new trial and must clear
  the same criteria plus a stricter bar for the number of trials so far.

## Trials so far (for multiple-testing awareness)
- Gold: London/PDH-PDL variants A-H, mean reversion I, daily trend J,
  250-day SMA K, Heikin Ashi swing (approx. 18 trials, all rejected).
- This test: trial 19, a different market and timeframe.
