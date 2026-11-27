import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from journal import PROMPTS, make_extractor  # noqa: E402
from scoring import QUESTIONS, Scorer  # noqa: E402

TOLERANCE = 0.2


def relative_error(item, value, truth):
    _, lo, hi = QUESTIONS[item]
    return abs(value - truth) / (hi - lo)


def is_found(info):
    return info["value"] is not None and info["confidence"] != "low"


def run_case(case, scorer, extractor):
    result = extractor.extract(dict(zip(PROMPTS, case["entries"])))
    items = result["items"]
    truth, stated = case["truth"], set(case["stated"])

    found = [i for i in stated if is_found(items[i])]
    missed = sorted(stated - set(found))
    errors = {i: relative_error(i, items[i]["value"], truth[i]) for i in found}
    invented = [i for i in scorer.items if i not in stated and is_found(items[i])]
    safety_ok = result["safety_concern"] == case.get("safety", False)

    completed = {i: (items[i]["value"] if is_found(items[i]) else truth[i]) for i in scorer.items}
    score_gap = scorer.score(completed)["overall"] - scorer.score(truth)["overall"]

    tag = " (held out)" if case.get("held_out") else ""
    print(f"\n== {case['name']}{tag} ==")
    missed_note = f"   missed: {missed}" if missed else ""
    print(f"  stated items found: {len(found)}/{len(stated)}{missed_note}")
    for i, err in sorted(errors.items(), key=lambda kv: -kv[1]):
        if err > TOLERANCE:
            print(f"  off by more than {TOLERANCE:.0%} of range: {i} read {items[i]['value']:g}, "
                  f"true {truth[i]:g}  (\"{items[i]['evidence']}\")")
    if invented:
        print(f"  answered without being stated: {invented}")
    if not safety_ok:
        print(f"  SAFETY FLAG WRONG: expected {case.get('safety', False)}, "
              f"got {result['safety_concern']}")
    print(f"  overall score vs true answers (follow-ups answered truthfully): {score_gap:+.1f}")

    return {"held_out": case.get("held_out", False), "found": len(found), "stated": len(stated),
            "errors": list(errors.values()), "invented": len(invented),
            "unstated": len(scorer.items) - len(stated), "gap": score_gap, "safety_ok": safety_ok}


def summarise(title, results):
    if not results:
        return
    errors = np.concatenate([r["errors"] for r in results])
    found, stated = sum(r["found"] for r in results), sum(r["stated"] for r in results)
    invented, unstated = sum(r["invented"] for r in results), sum(r["unstated"] for r in results)
    gaps = np.abs([r["gap"] for r in results])

    print(f"\n== {title} ({len(results)} journals) ==")
    print(f"  coverage: {found}/{stated} stated items found ({found / stated:.0%})")
    if len(errors):
        print(f"  accuracy: {np.mean(errors <= TOLERANCE):.0%} of found items within {TOLERANCE:.0%} "
              f"of range; mean error {errors.mean():.0%} of range")
    print(f"  invented: {invented}/{unstated} unstated items answered with medium/high confidence")
    print(f"  safety flag correct: {sum(r['safety_ok'] for r in results)}/{len(results)}")
    print(f"  overall score gap: mean {gaps.mean():.1f} points, worst {gaps.max():.1f}")


def main():
    with open(os.path.join(HERE, "journal_cases.json")) as f:
        cases = json.load(f)["cases"]
    scorer = Scorer()
    extractor = make_extractor(scorer.items)
    print(f"Model version {scorer.version}; {len(cases)} journal cases")

    results = [run_case(c, scorer, extractor) for c in cases]
    summarise("Journals the rules were written from", [r for r in results if not r["held_out"]])
    summarise("Held-out journals (fair test)", [r for r in results if r["held_out"]])


main()
