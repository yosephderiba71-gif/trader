# Heikin Ashi Swing Strategy - Frozen Rules

Status: frozen before implementation. These rules must not change after any results are seen.

## 1. Data source and preparation

- Source file: `data/XAUUSD_5m.csv`, loaded through `CsvMarketDataProvider` (`symbol="XAUUSD"`, `timeframe="5Min"`).
- The file actually covers `2023-01-02 00:00:00 UTC` through `2024-12-31 21:55:00 UTC` (175,649 five-minute bars). The task description said 2020-2026, but the file does not contain that span. All results are limited to the actual file.
- Remove synthetic flat padding bars where `open == high == low == close`. The feed uses these to pad closed-market periods; they are not real market observations and can create false Heikin Ashi signals. This is a data-quality step, not a strategy filter.
- Aggregate the remaining five-minute bars into one-hour bars using UTC hour boundaries:
  - `open` = first five-minute open in the hour
  - `high` = maximum five-minute high in the hour
  - `low` = minimum five-minute low in the hour
  - `close` = last five-minute close in the hour
  - `volume` = sum of five-minute volume in the hour
- Drop any hour that does not contain exactly 12 five-minute bars. No missing bars are filled.
- Use the cleaned one-hour bars only. No other filters, sessions, or resampling are applied.

## 2. Heikin Ashi calculation

For each closed one-hour bar, in chronological order:

- `HA_Close = (open + high + low + close) / 4`
- `HA_Open` for the first bar = `(open + close) / 2`
- `HA_Open` for every later bar = `(previous HA_Open + previous HA_Close) / 2`
- `HA_High = max(high, HA_Open, HA_Close)`
- `HA_Low = min(low, HA_Open, HA_Close)`

All calculations use full floating-point precision. Displayed values may be rounded, but signals and returns use unrounded values.

## 3. Signal definition

- Signals are evaluated only on closed one-hour Heikin Ashi bars.
- `body = HA_Close - HA_Open`
- `abs_body = abs(body)`
- `avg_body_10` = simple moving average of `abs_body` over the current bar and the previous nine bars (10-period average, including the signal bar). The first possible signal is therefore bar 10 (zero-based index 9).
- Long signal:
  - `body > 0` (green candle)
  - `HA_Open == HA_Low` (no bottom wick)
  - `abs_body > avg_body_10`
- Short signal:
  - `body < 0` (red candle)
  - `HA_Open == HA_High` (no top wick)
  - `abs_body > avg_body_10`
- A doji (`body == 0`) is never a signal.
- The equality tests are exact floating-point equality against the computed HA high/low.

## 4. Entry

- A signal on a closed one-hour bar fills at the open of the next one-hour bar. This prevents look-ahead bias.
- If the signal is on the final bar and no next bar exists, the signal is ignored.
- Entry price is the next bar's open.
- Only one position may be open at a time. Signals that occur while a position is open are ignored.
- The initial risk unit is `R = signal bar HA_High - signal bar HA_Low`.
- If `R <= 0`, the trade is skipped.
- There is no stop-loss order. `R` is used only to normalize profit and loss.

## 5. Exit

- Long exit: the first one-hour Heikin Ashi bar after entry with `HA_Close < HA_Open` (red candle). Fill at the next one-hour bar's open.
- Short exit: the first one-hour Heikin Ashi bar after entry with `HA_Close > HA_Open` (green candle). Fill at the next one-hour bar's open.
- The entry bar is eligible to be the exit-signal bar.
- A doji is not an opposite-color bar and does not trigger an exit.
- If an exit signal occurs on the final bar and no next bar exists, the position is marked to the final one-hour close for reporting only and flagged as `end_of_data`.
- If a position is still open at the end of the data, it is marked to the final one-hour close for reporting only and flagged as `end_of_data`.
- The strategy does not use a profit target, trailing stop, or time stop.

## 6. Costs

- Base case: `10` basis points per trade.
- Stress case: `20` basis points per trade.
- Cost in price terms = `entry_price * cost_bp / 10000`.
- Cost in R terms = `cost_price / R`.
- The cost is deducted once per trade and represents the total round-trip cost. No other commissions, spreads, swaps, or slippage are modeled.
- Gross R:
  - Long: `(exit_price - entry_price) / R`
  - Short: `(entry_price - exit_price) / R`
- Net R = `gross_R - cost_R`.

## 7. Position sizing

- Every trade is sized to exactly one unit of initial risk (`1R`).
- Returns are additive R multiples. There is no compounding and no portfolio-level position sizing.

## 8. In-sample / out-of-sample split

- Use the unique UTC calendar dates present in the cleaned one-hour bars, sorted ascending.
- The first 70% of whole days are in-sample.
- The last 30% of whole days are out-of-sample.
- Days are not split within a date. The in-sample set ends at the last bar of its last day; the out-of-sample set starts at the first bar of its first day.
- Heikin Ashi values and signal columns are computed on the full cleaned one-hour series using only past data. The trade simulator is then run independently on each split; no position is carried across the split boundary.
- A position still open at the end of a split is marked to that split's final one-hour close for reporting only and flagged as `end_of_data`.
- The out-of-sample section must not be opened, printed, or inspected until the rules file is complete and the in-sample run has been reviewed.

## 9. Metrics and reporting

- Report total trades, long trades, and short trades separately.
- Win rate: percentage of trades with net R greater than zero at the 10 bp base cost. Also report the gross win rate.
- Average R: mean gross R.
- Expectancy after costs: mean net R at 10 bp and at 20 bp.
- Worst single trade: minimum net R at 10 bp. Also report the worst gross R.
- Maximum drawdown: peak-to-trough decline of the cumulative net-R curve at 10 bp, in R. Also report the 20 bp value.
- Buy-and-hold out-of-sample return: `(last out-of-sample one-hour close / first out-of-sample one-hour close - 1) * 100`, using the same cleaned one-hour close-to-close prices. No costs are applied to buy-and-hold.
- Save the complete report as `research/ha_swing_report.json`.
- Print the out-of-sample section only after the in-sample section has been reviewed.

## 10. Freeze and no-tuning rule

- No parameter search, threshold search, alternative Heikin Ashi definitions, alternative exit rules, or data snooping is allowed.
- Do not change these rules after seeing any result.
- Report exactly what the data says, including negative or inconclusive results.
