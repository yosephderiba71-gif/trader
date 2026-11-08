# Discipline Bot: Cash-Flow Rebalancing & Dip Accelerator
STATUS: FROZEN.

## 1. Universe & Target Weights
- Assets: SPY (80%), TLT (20%).
- Execution: Standard Market Orders (fractional shares via `notional`) at 12:00 EST.

## 2. Cash-Flow Rebalancing
- Target Drift Band: ±3% absolute (SPY 77%-83%, TLT 17%-23%).
- Logic: Direct new cash 100% into the asset that is currently underweight.
- Explicit Sell Rebalance: Only trigger if an asset breaches the outer band (e.g., SPY > 83%) AND new cash is insufficient to fix it.

## 3. Market Dip Accelerator
- Base Deposit: $500 (configurable).
- If SPY is >15% below its 52-week high: Deposit = $625 (1.25x).
- If SPY is >25% below its 52-week high: Deposit = $750 (1.50x).

## 4. Idempotency & Safety
- State Check: Before placing an order, query Alpaca Activity API (`get_activities`). If an order has already filled this calendar month, abort.
- Hard Paper Assertion: Raise ValueError if `paper=True` is not explicitly set.
