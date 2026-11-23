import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from scoring import Scorer  # noqa: E402


def get(result, name):
    if name == "overall":
        return result["overall"]
    if name == "rest":
        return result["rest"]
    return result["subscores"][name]


def inside(value, bounds):
    return bounds[0] <= value <= bounds[1]


def load_json(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def run_profiles(scorer, spec, verbose):
    print("== 1. Hand-written profiles (overall score vs expected range) ==")
    passed = total = 0
    for p in spec["profiles"]:
        r = scorer.score(p["answers"])
        checks = [("overall", r["overall"], p["expected"])]
        checks += [(k, get(r, k), bounds) for k, bounds in p.get("subscores", {}).items()]
        fails = [(k, v, bounds) for k, v, bounds in checks if not inside(v, bounds)]
        total += len(checks)
        passed += len(checks) - len(fails)

        if verbose or fails:
            print(f"  {'PASS' if not fails else 'FAIL'}  {p['name']:<24} "
                  f"overall {r['overall']:5.1f}  expected {p['expected']}")
            for k, v, bounds in fails:
                if k != "overall":
                    print(f"        {k} = {v:.1f}, expected {bounds}")
            if verbose:
                subs = "  ".join(f"{k} {v:.0f}" for k, v in r["subscores"].items())
                print(f"        {subs}  rest {r['rest']:.0f}")
    print(f"  -> {passed}/{total} checks passed\n")


def run_direction_checks(scorer, spec, verbose):
    print("== 2. Direction checks (change one answer, does the score move the right way?) ==")
    profiles = {p["name"]: p for p in spec["profiles"]}
    passed = 0
    for c in spec["direction_checks"]:
        base = profiles[c["base"]]["answers"]
        before = get(scorer.score(base), c["score"])
        after = get(scorer.score({**base, **c["change"]}), c["score"])
        ok = after > before if c["expect"] == "up" else after < before
        passed += ok
        if verbose or not ok:
            print(f"  {'PASS' if ok else 'FAIL'}  {c['change']} -> {c['score']} "
                  f"{before:.1f} -> {after:.1f} (expected {c['expect']})")
    print(f"  -> {passed}/{len(spec['direction_checks'])} passed\n")


def run_real_cases(scorer, verbose):
    if not os.path.exists(os.path.join(HERE, "real_cases.json")):
        sys.exit("No real_cases.json yet: run python3 test_cases/make_real_cases.py first.")
    cases = load_json("real_cases.json")["cases"]

    print("== 3. Real held-out people (model score vs their dataset-score percentile) ==")
    predicted, actual, passed = [], [], 0
    for c in cases:
        v = scorer.score(c["answers"])["overall"]
        ok = inside(v, c["expected"])
        passed += ok
        predicted.append(v)
        actual.append(c["dataset_percentile"])
        if verbose or not ok:
            print(f"  {'PASS' if ok else 'FAIL'}  {c['name']:<16} model {v:5.1f}   "
                  f"dataset percentile {c['dataset_percentile']:5.1f}   expected {c['expected']}")

    predicted, actual = np.array(predicted), np.array(actual)
    err = np.abs(predicted - actual)
    print(f"  -> {passed}/{len(cases)} within range")
    print(f"     mean absolute error {err.mean():.1f} points, median {np.median(err):.1f}, "
          f"worst {err.max():.1f}")
    print(f"     correlation with dataset percentile {np.corrcoef(predicted, actual)[0, 1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scorer = Scorer()
    print(f"Model version {scorer.version}\n")

    spec = load_json("profiles.json")
    run_profiles(scorer, spec, args.verbose)
    run_direction_checks(scorer, spec, args.verbose)
    run_real_cases(scorer, args.verbose)


main()
