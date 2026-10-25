from trader.backtest import Metrics
from trader.config import Settings
from trader.research import Candidate, qualifying_best, report_frame

S = Settings(min_oos_trades=5, min_oos_sharpe=0.5)


def _cand(symbol, strategy, sharpe=1.0, ret=0.05, trades=10):
    return Candidate(symbol, strategy, Metrics(ret, sharpe, -0.05, trades, 0.5))


def test_best_qualifying_strategy_wins_per_symbol():
    best = qualifying_best([_cand("SPY", "a", sharpe=0.9), _cand("SPY", "b", sharpe=1.6)], S)
    assert best["SPY"].strategy == "b"


def test_too_few_trades_is_rejected():
    assert qualifying_best([_cand("SPY", "a", trades=4)], S) == {}


def test_negative_return_is_rejected_even_with_high_sharpe():
    assert qualifying_best([_cand("SPY", "a", sharpe=3.0, ret=-0.01)], S) == {}


def test_nan_sharpe_is_rejected():
    assert qualifying_best([_cand("SPY", "a", sharpe=float("nan"))], S) == {}


def test_report_frame_has_columns_even_when_empty():
    frame = report_frame([])
    assert "oos_sharpe" in frame.columns
    assert frame.empty
