import json
import os
import sys
import numpy as np

REPORT_PATH = "research/sector_momentum_backtest_report.json"
TRIAL_24_CALMAR = 0.82
OOS_YEARS = 6.5

def find_baseline_runs(data):
    candidate_keys = ["random_baseline", "clean_baseline", "baseline", "baseline_runs", "random_runs", "simulations"]
    for key in candidate_keys:
        if key in data and isinstance(data[key], list):
            return data[key]
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 50:
            return value
    return None

def extract_metric(run, metric_keys):
    for key in metric_keys:
        if key in run:
            return run[key]
    return None

def main():
    if not os.path.exists(REPORT_PATH):
        print(f"Error: Could not find report at {REPORT_PATH}")
        sys.exit(1)

    with open(REPORT_PATH, "r") as f:
        data = json.load(f)

    runs = find_baseline_runs(data)
    if not runs:
        print(f"Error: Could not locate baseline runs in JSON. Keys found: {list(data.keys())}")
        sys.exit(1)

    calmars = []
    for r in runs:
        c = extract_metric(r, ["calmar", "calmar_ratio", "Calmar"])
        if c is not None:
            calmars.append(float(c))
            continue

        ret_val = extract_metric(r, ["total_return", "return", "cum_return", "Total Return"])
        mdd_val = extract_metric(r, ["max_drawdown", "max_dd", "drawdown", "Max Drawdown"])

        if ret_val is None or mdd_val is None:
            continue

        ret = float(ret_val)
        mdd = abs(float(mdd_val))

        if ret > 2.0:
            ret = ret / 100.0
        if mdd > 1.0:
            mdd = mdd / 100.0

        if mdd <= 0:
            continue

        cagr = (1.0 + ret) ** (1.0 / OOS_YEARS) - 1.0
        calmar = cagr / mdd
        calmars.append(calmar)

    if not calmars:
        print("Error: Could not compute Calmar ratios from the provided data.")
        sys.exit(1)

    calmars = np.array(calmars)
    p95_calmar = np.percentile(calmars, 95)
    p90_calmar = np.percentile(calmars, 90)
    p50_calmar = np.percentile(calmars, 50)
    mean_calmar = np.mean(calmars)

    trial24_percentile = (calmars < TRIAL_24_CALMAR).mean() * 100.0
    passed = TRIAL_24_CALMAR >= p95_calmar

    print("\n" + "=" * 55)
    print("       TRIAL #26: CALMAR GAUNTLET VERIFICATION")
    print("=" * 55)
    print(f"Total Baseline Runs Analyzed : {len(calmars)}")
    print(f"Baseline Median (50th %ile)  : {p50_calmar:.4f}")
    print(f"Baseline Mean Calmar         : {mean_calmar:.4f}")
    print(f"Baseline 90th Percentile     : {p90_calmar:.4f}")
    print(f"Baseline 95th Percentile Bar : {p95_calmar:.4f}")
    print("-" * 55)
    print(f"Trial #24 Actual Calmar      : {TRIAL_24_CALMAR:.4f}")
    print(f"Trial #24 Percentile Rank    : {trial24_percentile:.2f}%")
    print("-" * 55)
    print(f"Gauntlet Pass Requirement    : >= 95.00%")
    print(f"Final Verdict                : {'PASSED (Greenlight)' if passed else 'FAILED (Rejected)'}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
