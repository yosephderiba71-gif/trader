# Intraday Meta-Strategy Backtest: frozen rules

Status: FROZEN. Do not edit. Any change after the run = a new trial, logged as such.

## Scope

- Test whether the intraday meta-strategy logic (select one asset + strategy over a lookback, then trade that combination for a fixed window) has a real edge or is overfitting.
- Simulator only. No broker, no live order path, no API keys.

## Frozen rules

1. Assets: SPY, QQQ, TSLA, AAPL, BTC-USD.
2. Timeframe: 15-minute bars (yfinance).
3. Lookback: 30 days of data to test all 20 combinations (4 strategies x 5 assets).
4. Strategies: MA Crossover, RSI, Bollinger Bands, VWAP.
5. Selection: pick the single asset+strategy combination with the highest net profit over the 30-day lookback.
6. Trading Window: trade ONLY that winning combination for the next 7 days.
7. Roll-forward: shift the window by 7 days and repeat until the end of the data (approx. 2022-2026).
8. Risk Management: 1% stop-loss, 2% take-profit.
9. Costs: apply 10 basis points (10bp) cost per trade.
10. Random Baseline: compare the final expectancy against 1,000 random strategies with the same trade count and holding times.
11. Strict constraints: do not tune any parameter; do not change the lookback (30 days) or trade window (7 days).
12. Output: JSON report showing total return, win rate, max drawdown, and a log of how often the bot switched strategies.

## Strategy definitions (frozen parameters)

- MA Crossover: SMA fast = 20 bars, SMA slow = 50 bars on 15-minute bars; long entry on fast crossing above slow, long exit on fast crossing below slow.
- RSI: Wilder RSI period = 14; long entry on RSI crossing above 30, long exit on RSI crossing below 70.
- Bollinger Bands: period = 20, multiplier = 2; long entry on close crossing above upper band, long exit on close crossing below middle band.
- VWAP: rolling VWAP period = 20 bars; EMA fast = 9 bars; long entry on EMA9 crossing above VWAP20, long exit on EMA9 crossing below VWAP20.

## Costs and fills

- Cost = 10bp per trade round trip; buy fill = open x (1 + cost/2), sell fill = open x (1 - cost/2).
- Entry fill = open of the next bar after the signal bar close; exit fill = open of the next bar after the exit signal bar close.
- Stop-loss = entry x (1 - 0.01); take-profit = entry x (1 + 0.02).

## If rejected

- Verdict: rejected / inconclusive. Log in REPORT.md. No retuning of thresholds, strategy parameters, lookback, or trade window.
- Any variant counts as a new trial and must clear the same criteria plus a stricter bar for the number of trials so far.
