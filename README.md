# Institutional Quantitative Trading & Research Engine

An end-to-end, institutional-grade quantitative research and execution pipeline built from scratch in Python. This repository demonstrates rigorous alpha research, fail-safe execution infrastructure, and a multi-agent AI development workflow.

## 🧠 1. Alpha Research Engine (27 Trials)
The core philosophy of this project is that **most retail strategies are overfitted noise**. We built a rigorous testing pipeline:
*   **Methodology:** 70/30 chronological out-of-sample splits, 10bp–20bp stress-test costs, and 1,000-run regime-matched random baselines.
*   **Results:** 27 strategy variants (RSI(2), Heikin Ashi, 11-sector momentum, etc.) were tested. All were rejected for failing to beat the 95th percentile of random chance, proving the Efficient Market Hypothesis on a retail budget.
*   **Artifacts:** The closest contender (Trial #24: 11-Sector Momentum) achieved a 144.5% OOS return with a -16.8% Max Drawdown (Calmar 0.86), but was rigorously rejected rather than deployed due to strict pre-registered rules.

## 🤖 2. Production Trading Bots (Paper Only)
*   **The Discipline Bot (Live):** An 80/20 SPY/TLT weekly rebalancer with a dip-accelerator (1.25x at >15% drawdown, 1.5x at >25%). Engineered with strict idempotency (ISO-week tags), non-margin cash guards, and DST-safe GitHub Actions cron jobs.
*   **Trial #24 Forward Test (Live):** A fully isolated monthly sector-momentum bot running on a dedicated paper account. Tests the Top 3 11-sector rotation against a SPY+Gate benchmark over a 3-5 year horizon.

## 🛠️ 3. Track B: Multi-Agent AI Development Workflow
We built a $0 local AI orchestration system to parallelize software development:
*   **Orchestrator:** Uses `tmux` and isolated `git worktrees` to spawn parallel coding agents without file collisions.
*   **Safety Gates:** Hard `pytest` spawn gates, strict `{"opencode", "claude"}` binary allowlists, and `--allow-dirty` protection.
*   **Memory & State:** A local SQLite database with `WAL` mode and `busy_timeout=5000` for concurrent agent memory, featuring regex secret-redaction and `chmod 600`.
*   **Reconciliation:** An independent `reconcile.py` script that queries Alpaca directly (with zero bot imports) to detect unauthorized trades and target weight drift.

## 📊 4. Proprietary GEX Data Collector
A custom Streamlit dashboard and GitHub Action that scrapes SPY options chains daily at 15:45 EST. It calculates Black-Scholes Gamma Exposure (GEX), Call/Put Walls, and the Gamma Flip level, automatically committing a proprietary historical CSV dataset to the repo.

## 🧪 Engineering Rigor
*   **301 passing unit tests** covering math, look-ahead bias, idempotency, and API failure states.
*   **Zero real-money risk:** Hard-coded `paper=True` assertions prevent live execution.

--------------------------------------------------
