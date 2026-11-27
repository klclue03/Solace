"""Test CLI: ask a user the survey questions (or a short journal), score them, save to SQLite.

    python3 model/score_cli.py                     # new user: questions; known user: past scores
    python3 model/score_cli.py --mode journal      # answer by journaling (analysed locally with spaCy)
    python3 model/score_cli.py --user alice        # skip the user-id prompt
    python3 model/score_cli.py --user alice --new  # known user answers the questions again
    python3 model/score_cli.py --user alice --answers answers.json   # no prompts
    python3 model/score_cli.py --history alice     # show saved results
"""
import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"))
from journal import PROMPTS, JournalError  # noqa: E402
from scoring import QUESTIONS  # noqa: E402
from service import DEFAULT_DB, PRIVACY_NOTICE, WellbeingService, label  # noqa: E402


def say(text):
    print("\n" + textwrap.fill(text, width=80))


def ask_one(service, item, prefix=""):
    question, lo, hi = QUESTIONS[item]
    while True:
        raw = input(f"{prefix}{question} ({lo}-{hi}): ").strip()
        try:
            return service.validate(item, raw)
        except ValueError:
            print(f"  Please enter a number from {lo} to {hi}.")


def ask(service):
    total = len(service.items)
    return {item: ask_one(service, item, f"[{number}/{total}] ")
            for number, item in enumerate(service.items, 1)}


def confirm(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def read_multiline(prompt):
    print(f"\n{prompt}\n  (write as much as you like; press Enter on an empty line to finish)")
    lines = []
    while True:
        line = input("  > ")
        if not line.strip():
            return "\n".join(lines)
        lines.append(line)


def extract_journal(service, entries):
    from spacy_extractor import SPACY_MODEL
    try:
        return service.extract_journal(entries)
    except ImportError:
        print("Journal mode needs spaCy: pip3 install spacy")
    except OSError:
        print(f"spaCy's English model is missing: python3 -m spacy download {SPACY_MODEL}")
    except JournalError as e:
        print(f"Sorry, that didn't work: {e}")
    return None


def review(service, extraction):
    items, found = extraction["items"], extraction["found"]
    answers = {i: items[i]["value"] for i in found}

    if found:
        print("\n== What we read from your journal ==")
        for number, item in enumerate(found, 1):
            info = items[item]
            print(f"  {number:>2}. {label(item):<18} {info['value']:g}  "
                  f"({info['confidence']})  \"{info['evidence']}\"")
        while True:
            choice = input("Type a number to change that answer, or press Enter to continue: ").strip()
            if not choice:
                break
            if not choice.isdigit() or not 1 <= int(choice) <= len(found):
                print(f"  Please enter a number from 1 to {len(found)}.")
                continue
            item = found[int(choice) - 1]
            answers[item] = ask_one(service, item, "  ")

    unclear = extraction["unclear"]
    if unclear:
        print(f"\nA few questions ({len(unclear)}) about things your journal didn't make clear:")
        for number, item in enumerate(unclear, 1):
            info = items[item]
            if info["value"] is not None:
                print(f"  We weren't sure, but guessed {info['value']:g} from: \"{info['evidence']}\"")
            answers[item] = ask_one(service, item, f"[{number}/{len(unclear)}] ")
    return answers


def journal_answers(service):
    say(PRIVACY_NOTICE)
    entries = {p: read_multiline(f"[{n}/{len(PROMPTS)}] {p}") for n, p in enumerate(PROMPTS, 1)}
    if not any(a.strip() for a in entries.values()):
        print("Your journal is empty.")
        return None

    print("\nReading your journal...")
    extraction = extract_journal(service, entries)
    if extraction is None:
        return None

    if extraction["support"]:
        say(extraction["support"]["message"])
        if not confirm("\nWould you like to continue to your results?"):
            sys.exit(0)

    return review(service, extraction), extraction


def delta(change):
    return "" if change is None else f"  ({change:+.1f})"


def show(result):
    changes = result["changes"] or {"overall": None, "subscores": {}, "rest": None}
    print("\n== Results ==")
    print(f"Overall wellbeing: {result['overall']:.1f} / 100{delta(changes['overall'])}")
    print("  (percentile: higher than this % of people in the training data)")
    print("Subscores (approximate percentiles):")
    for dim, value in result["subscores"].items():
        print(f"  {dim:<11} {value:5.1f}{delta(changes['subscores'].get(dim))}")
    print(f"  {'rest':<11} {result['rest']:5.1f}{delta(changes['rest'])}")
    if result["previous"]:
        print(f"Changes are compared with the previous entry from {result['previous']['created_at']}.")
    if result["support"]:
        say(result["support"]["message"])


def history(service, user):
    rows = service.history(user)
    if not rows:
        print(f"No saved results for user '{user}'.")
        return

    dims = list(rows[0]["subscores"])
    header = f"{'created_at':<26}{'source':<9}{'overall':>8}" + "".join(f"{d:>12}" for d in dims)
    print(header + f"{'rest':>8}  model")
    for row in rows:
        line = (f"{row['created_at']:<26}{row['source']:<9}{row['overall']:>8.1f}"
                + "".join(f"{row['subscores'].get(d, float('nan')):>12.1f}" for d in dims))
        print(line + f"{row['rest']:>8.1f}  {row['model_version']}")


def show_returning_user(service, user):
    print(f"\nWelcome back, {user}. Your past scores:\n")
    history(service, user)
    print("\nTo answer the questions again, run with --new "
          f"(e.g. python3 model/score_cli.py --user {user} --new).")


def load_model(service):
    try:
        service.scorer
    except (RuntimeError, FileNotFoundError) as e:
        sys.exit(f"Cannot load model: {e}")


def choose_mode(args):
    if args.answers:
        return "survey"
    if args.mode:
        return args.mode
    while True:
        choice = input("\nHow would you like to answer?\n"
                       "  1 = survey (18 quick number questions)\n"
                       "  2 = journal (5 open questions, written in your own words)\n"
                       "Choose 1 or 2: ").strip()
        if choice in ("1", "2"):
            return "survey" if choice == "1" else "journal"


def collect_answers(service, args):
    if args.answers:
        with open(args.answers) as f:
            return json.load(f), "survey", None

    if choose_mode(args) == "journal":
        result = journal_answers(service)
        if result is not None:
            answers, extraction = result
            return answers, "journal", extraction
        if not confirm("\nAnswer the survey instead?"):
            sys.exit(0)

    print(f"\nAnswer {len(service.items)} questions. Enter a whole or decimal number.\n")
    return ask(service), "survey", None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="user id (asked for if omitted)")
    ap.add_argument("--mode", choices=["survey", "journal"],
                    help="how to answer (asked for if omitted)")
    ap.add_argument("--answers", help="JSON file of {item: value} instead of prompting")
    ap.add_argument("--history", metavar="USER", help="print saved results for USER and exit")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file (default: {DEFAULT_DB})")
    ap.add_argument("--no-save", action="store_true", help="score without saving")
    ap.add_argument("--new", action="store_true",
                    help="answer the questions again even if the user already has results")
    return ap.parse_args()


def main():
    args = parse_args()
    service = WellbeingService(db_path=args.db)

    if args.history:
        history(service, args.history)
        return

    user = args.user or input("User ID: ").strip()
    if not user:
        sys.exit("A user ID is required.")

    if service.last_result(user) is not None and not (args.new or args.answers):
        show_returning_user(service, user)
        return

    load_model(service)
    answers, source, extraction = collect_answers(service, args)
    try:
        result = service.submit(user, answers, source, extraction, save=not args.no_save)
    except ValueError as e:
        sys.exit(f"Invalid answers: {e}")

    show(result)
    if not args.no_save:
        print(f"\nSaved for user '{user}' in {args.db}")


if __name__ == "__main__":
    main()
