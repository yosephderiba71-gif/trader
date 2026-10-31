import json

from trader.journal import load, record, summarize


def test_record_then_load_round_trip(tmp_path):
    path = tmp_path / "journal.jsonl"
    record(path, "skip", symbol="SPY", reason="no_qualifier")
    record(path, "trade_closed", symbol="GLD", ret=0.02, reason="target")

    events = load(path)
    assert [e["event"] for e in events] == ["skip", "trade_closed"]
    assert events[1]["ret"] == 0.02
    assert "ts" in events[0]


def test_record_creates_missing_folder(tmp_path):
    path = tmp_path / "nested" / "dir" / "journal.jsonl"
    record(path, "skip", reason="market_closed")
    assert path.exists()


def test_load_missing_file_is_empty(tmp_path):
    assert load(tmp_path / "nope.jsonl") == []


def test_load_skips_corrupt_and_blank_lines(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps({"event": "skip"}) + "\n"
        + "{not json\n"
        + "\n"
        + json.dumps([1, 2]) + "\n"          # valid JSON but not an event
        + json.dumps({"event": "trade_closed", "ret": 0.01}) + "\n",
        encoding="utf-8",
    )
    events = load(path)
    assert [e["event"] for e in events] == ["skip", "trade_closed"]


def test_summarize_counts_wins_and_reasons():
    events = [
        {"event": "trade_closed", "ret": 0.03, "reason": "target"},
        {"event": "trade_closed", "ret": -0.01, "reason": "stop"},
        {"event": "trade_closed", "ret": 0.02, "reason": "signal"},
        {"event": "skip", "reason": "no_qualifier"},
    ]
    out = summarize(events)
    assert out["trades"] == 3
    assert out["wins"] == 2
    assert out["losses"] == 1
    assert abs(out["win_rate"] - 2 / 3) < 1e-9
    assert out["exit_reasons"] == {"target": 1, "stop": 1, "signal": 1}
    assert out["skips"] == 1
    assert out["small_sample"] is True


def test_summarize_empty_history_is_safe():
    out = summarize([])
    assert out["trades"] == 0
    assert out["win_rate"] == 0.0
    assert out["avg_return"] == 0.0


def test_summarize_ignores_events_without_numeric_return():
    events = [
        {"event": "trade_closed", "ret": None, "reason": "stop"},
        {"event": "trade_closed", "ret": "0.5", "reason": "stop"},
        {"event": "trade_closed", "ret": 0.04, "reason": "target"},
    ]
    assert summarize(events)["trades"] == 1


def test_small_sample_flag_flips_at_thirty():
    events = [{"event": "trade_closed", "ret": 0.01, "reason": "target"}] * 30
    assert summarize(events)["small_sample"] is False
