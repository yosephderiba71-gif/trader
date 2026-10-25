from trader.config import Settings
from trader.risk import daily_loss_breached, plan_entry

S = Settings(risk_per_trade=0.01, max_position_pct=0.20, atr_stop_mult=2.0, atr_target_mult=3.0)


def test_sizing_respects_risk_budget_and_stop_and_target():
    # Risk budget 1,000. Stop distance 2 * 2 = 4, so risk allows 250 shares.
    # Position cap 20% of 100k at $100 allows 200 shares, which is the binding limit.
    plan = plan_entry(equity=100_000, price=100.0, atr_value=2.0, s=S)
    assert plan is not None
    assert plan.qty == 200
    assert plan.stop_price == 96.0
    assert plan.target_price == 106.0


def test_small_account_is_limited_by_risk_not_cap():
    plan = plan_entry(equity=1_000, price=100.0, atr_value=2.0, s=S)
    assert plan is not None
    assert plan.qty == 2  # risk budget 10 / stop distance 4 = 2 shares


def test_no_trade_when_one_share_breaks_the_limits():
    assert plan_entry(equity=100, price=100.0, atr_value=2.0, s=S) is None


def test_no_trade_on_bad_inputs():
    assert plan_entry(equity=100_000, price=100.0, atr_value=float("nan"), s=S) is None
    assert plan_entry(equity=100_000, price=100.0, atr_value=0.0, s=S) is None
    assert plan_entry(equity=0, price=100.0, atr_value=2.0, s=S) is None


def test_daily_loss_halt():
    assert daily_loss_breached(equity=97_000, last_equity=100_000, max_loss_pct=0.02) is True
    assert daily_loss_breached(equity=99_000, last_equity=100_000, max_loss_pct=0.02) is False
    assert daily_loss_breached(equity=97_000, last_equity=0, max_loss_pct=0.02) is False
