# Paper Trader v2

A long-only, risk-managed trading agent for Alpaca **paper** accounts. It runs as a
stateless cycle every 15 minutes on GitHub Actions. Alpaca is the single source of truth
for positions and orders.

## What changed from v1

| v1 problem | v2 fix |
|---|---|
| A SELL signal with no position **opened a short** | Long-only. Exit signals only close existing longs. |
| Fixed 1% stop and 2% target ignored volatility (hourly TSLA routinely moves more than 1%) | ATR-based stop and target (2× and 3× ATR) |
| Winning strategy chosen by in-sample dollar profit, which favoured high-priced assets such as BTC | Ranked by Sharpe on the **last 30% of history only**, with gates for min trades, positive return and min Sharpe |
| Only ~30 days of hourly data (about 150 bars per stock) | 180 days by default |
| VWAP was cumulative from the start of history, so it drifted meaninglessly | Rolling 20-bar VWAP with a volume filter |
| Crypto: fixed 0.01 quantity, no stop-loss, SELL rejected without a position | Crypto removed. Alpaca does not support bracket orders for crypto, so the position would be unprotected between runs |
| Could act on the still-forming bar | Forming bar dropped |
| Stop and target were DAY orders, so they expired at the close and left positions unprotected overnight | GTC bracket orders |
| No check for a re-entry on the same bar after a stop-out | Order-history check per bar, plus deterministic client order IDs |
| No daily loss limit | Entries halt after a 2% drop vs. the previous close. Exits still run. |
| Market orders sent when the market was closed | Checks the Alpaca market clock first |
| Dashboard keys hard-coded in the file | Environment variables only |
| Unpinned dependencies and no tests | Tests run before every trade. A failing test blocks the cycle. |

## Layout

```
trader/
  config.py      settings from environment variables
  strategies.py  4 causal, vectorised signal generators
  backtest.py    long-only backtester: next-bar fills, costs, intrabar stops
  research.py    out-of-sample strategy selection
  risk.py        position sizing and the daily loss halt (pure functions)
  agent.py       one cycle: exits, then entries (broker injected)
  broker.py      Alpaca orders and positions (the only write path)
  data.py        Alpaca bars, forming-bar removal
  cli.py         run / research / status
dashboard.py     read-only Streamlit view
tests/           37 tests; run with `python -m pytest`
.github/workflows/trade.yml
```

## Setup

1. Create a **paper** account at Alpaca and generate paper API keys.
2. Put the keys in the repo: Settings → Secrets and variables → Actions → `ALPACA_API_KEY`, `ALPACA_API_SECRET`. The old names `ALPACA_KEY` and `ALPACA_SECRET` also work locally, but the workflow reads the new names.
3. Run locally first:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   python -m pytest -q
   export ALPACA_API_KEY=... ALPACA_API_SECRET=...
   python -m trader research    # what the strategies did out-of-sample
   python -m trader run         # dry run: shows the decision, places nothing
   ```
4. Push to GitHub. The schedule starts on the next 15-minute mark during market hours.
5. Run `--execute` only after several days of dry-run logs look sensible. Use the Actions tab → trader → Run workflow → tick **execute**.

## Risk controls (defaults in `trader/config.py`)

- Risk 1% of equity per trade, measured from entry to the ATR stop.
- At most 20% of equity in one name and at most 3 open positions.
- At most 1 new entry per cycle.
- No new entries after a 2% drop versus the last close.
- Stop: entry − 2×ATR(14). Target: entry + 3×ATR(14). Both are server-side bracket legs.

## Limitations, read before trusting it

- **Not verified against the live Alpaca SDK in the environment where this was written.** The strategy, sizing, backtest and cycle logic are tested (37 tests). The `broker.py` and `data.py` files use alpaca-py APIs from memory, so run `python -m trader status` and `python -m trader run` on paper and check the logs before anything else.
- Strategy selection still carries selection bias. Out-of-sample testing reduces it but does not remove it. A backtest that looks good is a hypothesis, not evidence.
- Runs are 15 minutes apart and use hourly bars. This is a swing-style system. It does not react to intraday moves between runs; the bracket orders handle those.
- Scheduled workflows can start several minutes late, and GitHub may skip a run under load.
- GitHub pauses scheduled workflows in public repos after 60 days without activity. Push a small commit now and then.
- Free IEX data is a subset of market volume. Set `TRADER_FEED=sip` only if you have a paid data plan.
- Do not switch `paper=True` in `broker.py` to live. Going live needs a separate decision after a long paper record, and a new review of the code.

## Local dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard.py
```
