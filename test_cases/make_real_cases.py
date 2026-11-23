import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "model"))
import encoder  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tolerance", type=float, default=15)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def held_out_split():
    X, s, y = encoder.load(os.path.join(ROOT, "data", "Wellbeing_and_lifestyle_data_Kaggle.csv"))
    Xtr, Xte, s_tr, s_te, ytr, yte = train_test_split(X, s, y, test_size=0.2, random_state=0)
    _, _, _, _, ytr, _ = train_test_split(Xtr, s_tr, ytr, test_size=0.15, random_state=0)
    return Xte, s_te, ytr, yte


def build_case(i, raw_answers, Xte, s_te, ytr, yte, tolerance):
    answers = {c: float(raw_answers.iloc[i][c]) for c in encoder.FEATURES}
    answers[encoder.SLEEP] = float(s_te[i])
    pct = float((ytr < yte[i]).mean() * 100)
    return {
        "name": f"test_row_{int(Xte.index[i])}",
        "dataset_score": float(yte[i]),
        "dataset_percentile": round(pct, 1),
        "expected": [round(max(0, pct - tolerance), 1), round(min(100, pct + tolerance), 1)],
        "answers": answers,
    }


def main():
    args = parse_args()
    Xte, s_te, ytr, yte = held_out_split()

    raw_answers = Xte.copy()
    raw_answers[encoder.NEGATIVE] = -raw_answers[encoder.NEGATIVE]
    pick = np.random.default_rng(args.seed).choice(len(Xte), size=args.n, replace=False)
    cases = [build_case(i, raw_answers, Xte, s_te, ytr, yte, args.tolerance) for i in pick]

    out = os.path.join(HERE, "real_cases.json")
    with open(out, "w") as f:
        json.dump({"_note": f"{args.n} held-out test rows; expected = dataset percentile +/- {args.tolerance}",
                   "cases": cases}, f, indent=1)
    print(f"Wrote {len(cases)} cases to {out}")


main()
