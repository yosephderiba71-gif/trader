# Tasks

## ACTIVE: Place the research files from Downloads, verify them, and run the tests

This task is allowed to copy the files listed in step 2 into the repo and nothing else. REPORT.md is the only other file you may write. If any check in step 1 or step 3 fails, copy nothing, and report which check failed.

### Step 1: Check the sources in ~/Downloads

Run `ls ~/Downloads` and record every file whose name contains "orb", "research", or "test_orb". If any name appears twice, such as "orb (1).py" next to "orb.py", stop and report it.

Then run each check and record the output. Each must print the stated result:

- `grep -c "def run_orb" ~/Downloads/orb.py`, expect 1 or more
- `grep -c "trail" ~/Downloads/orb.py`, expect 1 or more
- `grep -c "test_trailing_stop_follows_prior_low" ~/Downloads/test_orb.py`, expect 1
- `grep -c "def evaluate" ~/Downloads/run_orb_research.py`, expect 1 or more
- `grep -c "test_research_module_cannot_place_orders" ~/Downloads/test_orb_research.py`, expect 1
- `grep -c "if: always" ~/Downloads/research.yml`, expect 1

### Step 2: Copy the files (only if every check in step 1 passed)

From `~/Desktop/trading-agent`, run:

```
mkdir -p research tests trader .github/workflows
touch research/__init__.py
cp ~/Downloads/run_orb_research.py research/run_orb_research.py
cp ~/Downloads/test_orb_research.py tests/test_orb_research.py
cp ~/Downloads/orb.py trader/orb.py
cp ~/Downloads/test_orb.py tests/test_orb.py
cp ~/Downloads/research.yml .github/workflows/research.yml
```

Do not copy any other file.

### Step 3: Check the copies in the repo

Run the same six checks from step 1, but on the repo paths instead of ~/Downloads, for example `grep -c "def run_orb" trader/orb.py`. Each must print the same expected result.

### Step 4: Run the tests and check the packages

- `/Users/yosi/venvs/quantstats/bin/python -m pytest -q`, with the full output
- `/Users/yosi/venvs/quantstats/bin/python -m pip show alpaca-py pandas numpy pytest`, with the version lines only. Do not install anything.

### Step 5: Write REPORT.md

Include, in order: the Downloads listing, each check result from step 1, each copy command you ran, each check result from step 3, the full pytest output, and the package versions. Add a line for anything that looks wrong. Expected pytest result: 74 passed. If the number differs, report it; do not change any file to fix it.

## Done

- The user has already copied AGENTS.md and TASKS.md into the repo root.
