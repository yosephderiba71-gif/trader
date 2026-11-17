from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pytest

from dev_tools import orchestrator


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize("agent", ["opencode", "claude"])
def test_validate_agent_accepts_allowlisted_agents(agent):
    assert orchestrator.validate_agent(agent) == agent


@pytest.mark.parametrize("agent", ["agy", "codex", "", "OpenCode"])
def test_validate_agent_rejects_everything_else(agent):
    with pytest.raises(ValueError, match="Unsupported agent"):
        orchestrator.validate_agent(agent)


def test_main_rejects_unsupported_agent():
    with pytest.raises(ValueError, match="Unsupported agent"):
        orchestrator.main(["--agent", "agy"])


def test_dirty_repo_aborts_before_spawning(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        orchestrator,
        "main_repo_is_dirty",
        lambda repo_root, run=None: True,
    )

    spawned = orchestrator.spawn_workers(
        "opencode",
        workers=2,
        repo_root=tmp_path,
        memory_db=tmp_path / "memory.db",
        rate_limit_path=tmp_path / "rate_limits.json",
    )

    assert spawned == []
    assert capsys.readouterr().out.strip() == orchestrator.DIRTY_REPO_MESSAGE
    assert not (tmp_path / "memory.db").exists()
    assert not (tmp_path / "rate_limits.json").exists()


def test_log_message_redacts_api_keys_in_fts5_table(tmp_path):
    db_path = tmp_path / "memory.db"
    api_key = "AKIA" + "Z" * 16

    orchestrator.log_message(
        "opencode",
        f"worker used {api_key} for the call",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT timestamp, agent_name, message FROM agent_logs"
        ).fetchone()
        table = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'agent_logs'"
        ).fetchone()

    assert table == ("agent_logs",)
    assert row[0]
    assert row[1] == "opencode"
    assert "[REDACTED]" in row[2]
    assert api_key not in row[2]


def test_scrub_secrets_replaces_long_uppercase_tokens():
    assert orchestrator.scrub_secrets("token ABCDEFGHIJKLMNOPQRST done") == (
        "token [REDACTED] done"
    )


def test_create_worktree_removes_env_file(tmp_path):
    repo = tmp_path / "trading-agent"
    repo.mkdir()
    worktree = tmp_path / "trading-agent-w1"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        worktree.mkdir()
        (worktree / ".env").write_text("SECRET=value")
        return _completed()

    created = orchestrator.create_worktree(repo, 1, run=fake_run)

    assert created == worktree
    assert calls[0][:5] == ["git", "-C", str(repo.resolve()), "worktree", "add"]
    assert not (worktree / ".env").exists()


def test_rate_limit_sleeps_before_fourth_spawn(tmp_path):
    rate_limit_path = tmp_path / "rate_limits.json"
    clock = [1000.0]
    sleeps = []

    def now():
        return clock[0]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    for _ in range(3):
        orchestrator.wait_for_rate_limit(
            rate_limit_path,
            now_fn=now,
            sleep_fn=sleep,
        )

    assert sleeps == []

    orchestrator.wait_for_rate_limit(
        rate_limit_path,
        now_fn=now,
        sleep_fn=sleep,
    )

    assert len(sleeps) == 1
    assert sleeps[0] >= orchestrator.RATE_LIMIT_WINDOW_SECONDS
    assert len(orchestrator.load_spawn_timestamps(rate_limit_path)) == 1


def test_tmux_command_wraps_worker_in_caffeinate(tmp_path):
    command = orchestrator.tmux_worker_command(
        session="session",
        window="w1",
        worktree=tmp_path,
        command="opencode",
        create_session=True,
    )

    assert command[:6] == ["tmux", "new-session", "-d", "-s", "session", "-n"]
    assert command[-1] == "caffeinate -i opencode"


def test_cleanup_kills_sessions_removes_worktrees_and_prunes(tmp_path):
    repo = tmp_path / "trading-agent"
    repo.mkdir()
    worktree = tmp_path / "trading-agent-w1"
    worktree.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["tmux", "list-sessions"]:
            return _completed(stdout="trading-agent-orchestrator-1\nother\n")
        if command[:5] == ["git", "-C", str(repo.resolve()), "worktree", "list"]:
            return _completed(
                stdout=f"worktree {repo.resolve()}\nworktree {worktree}\n"
            )
        return _completed()

    assert orchestrator.cleanup(repo_root=repo, run=fake_run) == 0
    assert ["tmux", "kill-session", "-t", "trading-agent-orchestrator-1"] in calls
    assert [
        "git",
        "-C",
        str(repo.resolve()),
        "worktree",
        "remove",
        "--force",
        str(worktree),
    ] in calls
    assert ["git", "-C", str(repo.resolve()), "worktree", "prune"] in calls
    assert not worktree.exists()
