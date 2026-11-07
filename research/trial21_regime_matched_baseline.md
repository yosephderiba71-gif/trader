# Trial #21: Regime-Matched Random Baseline (RSI(2) on SPY/QQQ)

STATUS: PRE-REGISTERED. Do not edit after the first run.

## Hypothesis
H1: RSI(2) timing adds value beyond simply being long in an uptrend.
H0: RSI(2) OOS expectancy is indistinguishable from random entries that are also restricted to close > SMA200(t).

## What is changing vs Trial #19
Only the random baseline generation. The RSI(2) strategy rules are FROZEN:
- Entry: close > SMA200(t) AND RSI(2) < 10. Buy next open.
- Exit: close > SMA5(t) OR 10 trading days. Sell next open.
- Costs: 5bp base / 10bp stress.

## Baseline requirements (must all be true)
1. 1,000 runs.
2. Each run generates N random entry days, where N = the real OOS trade count.
3. Every sampled day MUST satisfy close > SMA200(t).
4. Sampling is WITHOUT replacement.
5. Holding periods are sampled from the real OOS holding-period distribution, also without replacement.
6. No overlapping holding windows within a single run.
7. Entry and exit fills use next-day open, identical to the real strategy.
8. Costs are applied identically.

## Statistical reporting (must output all of these)
- Mean, median, 5th percentile, 95th percentile, and standard deviation of the 1,000 baseline expectancies.
- The real strategy's OOS expectancy.
- One-sided p-value = fraction of baseline runs with expectancy >= real expectancy.
- The full 1,000-value array saved to JSON.

## Pass criteria (frozen, checked on OOS only)
P1. OOS expectancy after 10bp stress > 0%.
P2. OOS expectancy after 5bp base >= 0.20%.
P3. Full-history trade count >= 100.
P4 (new). OOS expectancy > 95th percentile of regime-matched baseline AND one-sided p-value <= 0.05.
P5. Expectancy positive in both chronological halves.
P6. OOS max drawdown no worse than -20%.
P7. Positive expectancy in >= 60% of calendar years with trades.

## Decision rules (pre-committed)
- All seven true -> PASS. Proceed to forward paper trial.
- P4 false -> REJECT. Log and stop. No retuning.
- Any other criterion false -> REJECT. Log and stop.
