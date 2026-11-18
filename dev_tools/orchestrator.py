from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ALLOWED_AGENTS = frozenset({"opencode", "claude"})
DIRTY_REPO_MESSAGE = "Error: Main repo is dirty. Commit changes first."
DIRTY_REPO_WARNING = (
    "Warning: Main repo has uncommitted changes. git worktree add branches "
    "from HEAD and will not include them; continuing because --allow-dirty "
    "was given."
)
TESTS_FAILING_MESSAGE = (
    "Error: Tests on main are failing. Fix them before spawning workers."
)
PYTEST_PYTHON = "/Users/yosi/venvs/quantstats/bin/python"
PYTEST_COMMAND = (PYTEST_PYTHON, "-m", "pytest", "-q")
SQLITE_BUSY_TIMEOUT_MS = 5000
MEMORY_DB_FILE_MODE = 0o600
RATE_LIMIT_MAX_SPAWNS = 3
RATE_LIMIT_WINDOW_SECONDS = 300.0
DEFAULT_WORKERS = 2
API_KEY_PATTERN = re.compile(r"\b[A-Z0-9]{20,}\b")

DEV_TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = DEV_TOOLS_DIR.parent
DEFAULT_MEMORY_DB = DEV_TOOLS_DIR / "memory.db"
DEFAULT_RATE_LIMIT_PATH = DEV_TOOLS_DIR / "rate_limits.json"


def validate_agent(agent: str) -> str:
    if agent not in ALLOWED_AGENTS:
        raise ValueError(
            f"Unsupported agent {agent!r}. Allowed agents: opencode, claude"
        )
    return agent


def scrub_secrets(message: str) -> str:
    return API_KEY_PATTERN.sub("[REDACTED]", message)


def connect_memory_db(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(db_path),
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
    )
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def secure_memory_db(db_path: str | Path) -> None:
    path = Path(db_path)
    for candidate in (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    ):
        if candidate.exists():
            candidate.chmod(MEMORY_DB_FILE_MODE)


def init_memory_db(db_path: str | Path = DEFAULT_MEMORY_DB) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_memory_db(path)
    try:
        with connection:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS agent_logs "
                "USING fts5(timestamp, agent_name, message)"
            )
    except sqlite3.OperationalError as exc:
        raise RuntimeError("SQLite FTS5 support is required for memory.db") from exc
    finally:
        connection.close()
    secure_memory_db(path)
    return path


def log_message(
    agent_name: str,
    message: str,
    db_path: str | Path = DEFAULT_MEMORY_DB,
) -> None:
    init_memory_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    safe_message = scrub_secrets(message)
    connection = connect_memory_db(db_path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO agent_logs(timestamp, agent_name, message) VALUES (?, ?, ?)",
                (timestamp, agent_name, safe_message),
            )
    finally:
        connection.close()
    secure_memory_db(Path(db_path))


def load_spawn_timestamps(
    rate_limit_path: str | Path = DEFAULT_RATE_LIMIT_PATH,
) -> list[float]:
    path = Path(rate_limit_path)
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if isinstance(raw, dict):
        raw = raw.get("spawn_timestamps", [])
    if not isinstance(raw, list):
        return []
    return [float(value) for value in raw if isinstance(value, (int, float))]


def save_spawn_timestamps(
    timestamps: list[float],
    rate_limit_path: str | Path = DEFAULT_RATE_LIMIT_PATH,
) -> None:
    path = Path(rate_limit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps({"spawn_timestamps": timestamps}, indent=2))
    temporary.replace(path)


def wait_for_rate_limit(
    rate_limit_path: str | Path = DEFAULT_RATE_LIMIT_PATH,
    max_spawns: int = RATE_LIMIT_MAX_SPAWNS,
    window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
    now_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> float:
    if max_spawns < 1:
        raise ValueError("max_spawns must be at least 1")

    now_fn = now_fn or time.time
    sleep_fn = sleep_fn or time.sleep
    timestamps = load_spawn_timestamps(rate_limit_path)
    now = now_fn()
    timestamps = [stamp for stamp in timestamps if 0 <= now - stamp < window_seconds]

    while len(timestamps) >= max_spawns:
        delay = max(0.05, window_seconds - (now - min(timestamps)))
        sleep_fn(delay)
        now = now_fn()
        timestamps = [
            stamp for stamp in timestamps if 0 <= now - stamp < window_seconds
        ]

    timestamps.append(now)
    save_spawn_timestamps(timestamps, rate_limit_path)
    return now


def _run(
    command: list[str],
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    runner = run or subprocess.run
    kwargs: dict[str, object] = {
        "capture_output": True,
        "text": True,
        "check": False,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    try:
        return runner(command, **kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {command[0]}") from exc


def tests_pass_on_main(
    repo_root: str | Path,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> bool:
    repo = Path(repo_root).resolve()
    result = _run(list(PYTEST_COMMAND), run=run, cwd=repo)
    return result.returncode == 0


def main_repo_is_dirty(
    repo_root: str | Path,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> bool:
    repo = Path(repo_root).resolve()
    result = _run(["git", "-C", str(repo), "status", "--porcelain"], run=run)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    return bool(result.stdout.strip())


def ensure_clean_repo(
    repo_root: str | Path,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    allow_dirty: bool = False,
) -> bool:
    if main_repo_is_dirty(repo_root, run=run):
        if allow_dirty:
            print(DIRTY_REPO_WARNING)
            return True
        print(DIRTY_REPO_MESSAGE)
        return False
    return True


def worktree_path(repo_root: str | Path, worker_index: int) -> Path:
    if worker_index < 1:
        raise ValueError("worker_index must be at least 1")
    repo = Path(repo_root).resolve()
    return repo.parent / f"{repo.name}-w{worker_index}"


def enforce_no_env_file(worktree: str | Path) -> None:
    path = Path(worktree)
    env_file = path / ".env"
    if env_file.is_symlink() or env_file.is_file():
        env_file.unlink()
    elif env_file.exists():
        raise RuntimeError(f"Refusing to leave a non-file .env at {env_file}")


def create_worktree(
    repo_root: str | Path,
    worker_index: int,
    base_ref: str = "HEAD",
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> Path:
    repo = Path(repo_root).resolve()
    path = worktree_path(repo, worker_index)
    if path.exists():
        raise RuntimeError(f"Worktree path already exists: {path}")

    command = [
        "git",
        "-C",
        str(repo),
        "worktree",
        "add",
        "--detach",
        str(path),
        base_ref,
    ]
    result = _run(command, run=run)
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    enforce_no_env_file(path)
    return path


def wrap_with_caffeinate(command: str) -> str:
    return f"caffeinate -i {command}"


def tmux_worker_command(
    session: str,
    window: str,
    worktree: str | Path,
    command: str,
    create_session: bool,
) -> list[str]:
    wrapped = wrap_with_caffeinate(command)
    if create_session:
        return [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-n",
            window,
            "-c",
            str(worktree),
            wrapped,
        ]
    return [
        "tmux",
        "new-window",
        "-t",
        session,
        "-n",
        window,
        "-c",
        str(worktree),
        wrapped,
    ]


def spawn_workers(
    agent: str,
    workers: int = DEFAULT_WORKERS,
    command: str | None = None,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    memory_db: str | Path = DEFAULT_MEMORY_DB,
    rate_limit_path: str | Path = DEFAULT_RATE_LIMIT_PATH,
    session_prefix: str | None = None,
    allow_dirty: bool = False,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> list[Path]:
    validate_agent(agent)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    repo = Path(repo_root).resolve()
    if not tests_pass_on_main(repo, run=run):
        print(TESTS_FAILING_MESSAGE)
        return []
    if not ensure_clean_repo(repo, run=run, allow_dirty=allow_dirty):
        return []

    init_memory_db(memory_db)
    worker_command = command or agent
    prefix = session_prefix or f"{repo.name}-orchestrator"
    session = f"{prefix}-{int(time.time() * 1000)}"
    spawned: list[Path] = []

    for index in range(1, workers + 1):
        wait_for_rate_limit(rate_limit_path)
        path = create_worktree(repo, index, run=run)
        tmux_command = tmux_worker_command(
            session=session,
            window=f"w{index}",
            worktree=path,
            command=worker_command,
            create_session=index == 1,
        )
        result = _run(tmux_command, run=run)
        if result.returncode != 0:
            raise RuntimeError(f"tmux failed: {result.stderr.strip()}")

        spawned.append(path)
        message = (
            f"Spawned {agent} worker {index} at {path} "
            f"in tmux session {session}"
        )
        log_message(agent, message, db_path=memory_db)
        print(message)

    return spawned


def list_tmux_sessions(
    prefix: str,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> list[str]:
    try:
        result = _run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            run=run,
        )
    except RuntimeError:
        return []
    if result.returncode != 0:
        return []
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith(prefix)
    ]


def list_worktree_paths(
    repo_root: str | Path,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> list[Path]:
    repo = Path(repo_root).resolve()
    result = _run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        run=run,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git worktree list failed: {result.stderr.strip()}")
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            paths.append(Path(line.removeprefix("worktree ")))
    return paths


def cleanup(
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    session_prefix: str | None = None,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> int:
    repo = Path(repo_root).resolve()
    prefix = session_prefix or f"{repo.name}-orchestrator"

    sessions = list_tmux_sessions(prefix, run=run)
    for session in sessions:
        _run(["tmux", "kill-session", "-t", session], run=run)

    removed: list[Path] = []
    for path in list_worktree_paths(repo, run=run):
        if path.parent.resolve() != repo.parent:
            continue
        if not path.name.startswith(f"{repo.name}-w"):
            continue
        _run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(path)],
            run=run,
        )
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        removed.append(path)

    _run(["git", "-C", str(repo), "worktree", "prune"], run=run)
    print(
        f"Cleanup complete: killed {len(sessions)} tmux session(s), "
        f"removed {len(removed)} worktree(s)."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spawn isolated coding agents in git worktrees and tmux."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("spawn", "cleanup"),
        default="spawn",
    )
    parser.add_argument("--agent", help="Worker agent: opencode or claude")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of workers to spawn (default: 2)",
    )
    parser.add_argument(
        "--command",
        dest="worker_command",
        help="Shell command to run inside each worker (default: the agent name)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Kill tmux sessions and remove worker worktrees",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Proceed even if the main repo has uncommitted changes "
            "(pytest must still pass)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cleanup or args.command == "cleanup":
        return cleanup()

    if not args.agent:
        raise ValueError("--agent is required for spawn")

    spawned = spawn_workers(
        agent=args.agent,
        workers=args.workers,
        command=args.worker_command,
        allow_dirty=args.allow_dirty,
    )
    return 0 if spawned else 1


if __name__ == "__main__":
    raise SystemExit(main())
