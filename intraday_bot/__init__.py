"""SPY intraday-momentum paper bot, isolated from the existing swing bot.

The strategy is a long-only adaptation of Zarattini, Aziz & Barbon (2024),
"Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF".
It is paper-only. It never touches ``trader/`` or the live workflow.
"""

__version__ = "0.1.0"
