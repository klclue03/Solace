"""Test CLI: ask a user the survey questions, score them, save to SQLite.

    python3 model/score_cli.py                     # new user: questions; known user: past scores
    python3 model/score_cli.py --user alice        # skip the user-id prompt
    python3 model/score_cli.py --user alice --new  # known user answers the questions again
    python3 model/score_cli.py --user alice --answers answers.json   # no prompts
    python3 model/score_cli.py --history alice     # show saved results
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

from scoring import HERE, QUESTIONS, Scorer

DEFAULT_DB = os.path.join(os.path.dirname(HERE), "data", "responses.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    answers       TEXT NOT NULL,
    subscores     TEXT NOT NULL,
    rest          REAL NOT NULL,
    overall       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses (user_id, created_at);
"""


def connect(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def ask(scorer):
    answers = {}
    total = len(scorer.items)
    for number, item in enumerate(scorer.items, 1):
        question, lo, hi = QUESTIONS[item]
        while True:
            raw = input(f"[{number}/{total}] {question} ({lo}-{hi}): ").strip()
            try:
                answers[item] = scorer.validate(item, raw)
                break
            except ValueError:
                print(f"  Please enter a number from {lo} to {hi}.")
    return answers


def last_result(con, user):
    row = con.execute("SELECT overall, subscores, rest, created_at FROM responses "
                      "WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                      (user,)).fetchone()
    if row is None:
        return None
    overall, subscores, rest, at = row
    return {"overall": overall, "subscores": json.loads(subscores), "rest": rest, "at": at}


def delta(new, old):
    return "" if old is None else f"  ({new - old:+.1f})"


def show(result, prev):
    print("\n== Results ==")
    print(f"Overall wellbeing: {result['overall']:.1f} / 100"
          f"{delta(result['overall'], prev and prev['overall'])}")
    print("  (percentile: higher than this % of people in the training data)")
    print("Subscores (approximate percentiles):")
    for dim, value in result["subscores"].items():
        print(f"  {dim:<11} {value:5.1f}{delta(value, prev and prev['subscores'].get(dim))}")
    print(f"  {'rest':<11} {result['rest']:5.1f}{delta(result['rest'], prev and prev['rest'])}")
    if prev:
        print(f"Changes are compared with the previous entry from {prev['at']}.")


def save(con, user, result):
    con.execute("INSERT INTO responses (user_id, created_at, model_version, answers, "
                "subscores, rest, overall) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user,
                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 result["model_version"],
                 json.dumps(result["answers"]),
                 json.dumps(result["subscores"]),
                 result["rest"],
                 result["overall"]))
    con.commit()


def history(con, user):
    rows = con.execute("SELECT created_at, overall, rest, subscores, model_version "
                       "FROM responses WHERE user_id = ? ORDER BY created_at, id",
                       (user,)).fetchall()
    if not rows:
        print(f"No saved results for user '{user}'.")
        return

    dims = list(json.loads(rows[0][3]))
    header = f"{'created_at':<26}{'overall':>8}" + "".join(f"{d:>12}" for d in dims)
    print(header + f"{'rest':>8}  model")
    for at, overall, rest, subscores, version in rows:
        subs = json.loads(subscores)
        line = f"{at:<26}{overall:>8.1f}" + "".join(f"{subs.get(d, float('nan')):>12.1f}" for d in dims)
        print(line + f"{rest:>8.1f}  {version}")


def show_returning_user(con, user):
    print(f"\nWelcome back, {user}. Your past scores:\n")
    history(con, user)
    print("\nTo answer the questions again, run with --new "
          f"(e.g. python3 model/score_cli.py --user {user} --new).")


def load_scorer():
    try:
        return Scorer()
    except (RuntimeError, FileNotFoundError) as e:
        sys.exit(f"Cannot load model: {e}")


def collect_answers(scorer, answers_file):
    if answers_file:
        with open(answers_file) as f:
            return json.load(f)
    print(f"Answer {len(scorer.items)} questions. Enter a whole or decimal number.\n")
    return ask(scorer)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="user id (asked for if omitted)")
    ap.add_argument("--answers", help="JSON file of {item: value} instead of prompting")
    ap.add_argument("--history", metavar="USER", help="print saved results for USER and exit")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file (default: {DEFAULT_DB})")
    ap.add_argument("--no-save", action="store_true", help="score without saving")
    ap.add_argument("--new", action="store_true",
                    help="answer the questions again even if the user already has results")
    return ap.parse_args()


def main():
    args = parse_args()
    con = connect(args.db)

    if args.history:
        history(con, args.history)
        return

    user = args.user or input("User ID: ").strip()
    if not user:
        sys.exit("A user ID is required.")

    is_returning = last_result(con, user) is not None
    if is_returning and not (args.new or args.answers):
        show_returning_user(con, user)
        return

    scorer = load_scorer()
    answers = collect_answers(scorer, args.answers)
    try:
        result = scorer.score(answers)
    except ValueError as e:
        sys.exit(f"Invalid answers: {e}")

    show(result, last_result(con, user))
    if not args.no_save:
        save(con, user, result)
        print(f"\nSaved for user '{user}' in {args.db}")


if __name__ == "__main__":
    main()
