import json

from trader.agent import log
from trader.journal import load


def test_log_writes_to_journal_when_env_set(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trades.jsonl"
    monkeypatch.setenv("TRADER_JOURNAL", str(path))
    log("skip", symbol="SPY", reason="position_too_small")

    events = load(path)
    assert events[0]["event"] == "skip"
    assert events[0]["symbol"] == "SPY"
    # stdout still gets the same event, so the Actions log is unchanged
    assert json.loads(capsys.readouterr().out)["event"] == "skip"


def test_log_writes_no_file_when_env_unset(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TRADER_JOURNAL", raising=False)
    monkeypatch.chdir(tmp_path)
    log("skip", symbol="SPY", reason="market_closed")
    assert list(tmp_path.iterdir()) == []


def test_journal_failure_does_not_raise(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file-not-folder"
    blocker.write_text("x")  # a file where the journal's folder should be
    monkeypatch.setenv("TRADER_JOURNAL", str(blocker / "trades.jsonl"))
    log("skip", symbol="SPY", reason="market_closed")  # must not raise
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert lines[-1]["event"] == "journal_error"
