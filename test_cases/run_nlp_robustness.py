import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
from journal import PROMPTS, make_extractor  # noqa: E402
from scoring import Scorer  # noqa: E402
from spacy_rules import donation_scale  # noqa: E402

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten"}

TEMPLATES = {
    "SLEEP_HOURS": (["I sleep {d} hours a night.", "I usually get about {w} hours of sleep.",
                     "Sleep: maybe {d} hours.", "Most nights I only manage {w} hours of sleep."],
                    [4, 6, 7, 9], None),
    "DAILY_STEPS": (["I walk around {d},000 steps a day.", "I do about {d}k steps.",
                     "My phone says I average {d},000 steps daily."], [2, 5, 8], None),
    "SOCIAL_NETWORK": (["I talk to about {d} people a day.", "I see roughly {d} people each day.",
                        "I interact with {w} people on a normal day."], [2, 5, 8], None),
    "CORE_CIRCLE": (["I have {w} close friends.", "There are {d} people I'm really close to.",
                     "I trust {w} people completely."], [1, 3, 6], None),
    "FRUITS_VEGGIES": (["I eat {d} portions of fruit and veg a day.",
                        "I have {w} servings of vegetables daily."], [1, 3, 5], None),
    "FLOW": (["I'm fully focused for about {d} hours a day.",
              "I get absorbed in my work for {w} hours most days."], [2, 4, 6], None),
    "TIME_FOR_PASSION": (["I spend {d} hours a day on my hobbies.",
                          "My passion is music, about {w} hours a day."], [1, 2, 4], None),
    "DAILY_SHOUTING": (["I shout at someone about {w} times a week.",
                        "I lose my temper {d} times a week."], [1, 3, 5], None),
    "WEEKLY_MEDITATION": (["I meditate {d} times a week.",
                           "I take time to reflect {w} times a week."], [1, 3, 5], None),
    "PLACES_VISITED": (["I visited {d} new places this year.",
                        "This year I travelled to {w} new cities."], [1, 3, 6], None),
    "LOST_VACATION": (["I left {d} days of vacation unused last year.",
                       "I didn't use {w} days of my holiday this year."], [2, 5, 8], None),
    "SUPPORTING_OTHERS": (["I help {d} people regularly.", "I mentor {w} people at work."],
                          [1, 3, 5], None),
    "DONATION": (["I donate to charity {d} times a year.",
                  "I volunteer about {w} times a year."], [1, 3, 8], donation_scale),
    "LIVE_VISION": (["My goals are clear for the next {d} years.",
                     "I have a plan for the next {w} years."], [1, 3, 5], None),
}

LEVEL_PHRASES = [
    ("DAILY_STRESS", "I'm not stressed at all.", 0),
    ("DAILY_STRESS", "I'm a little stressed.", 2),
    ("DAILY_STRESS", "I'm pretty stressed.", 3),
    ("DAILY_STRESS", "I'm very stressed.", 4),
    ("DAILY_STRESS", "I'm extremely stressed.", 5),
    ("DAILY_STRESS", "Work stresses me out a lot.", 4),
    ("DAILY_STRESS", "I feel calm most days.", 1),
    ("DAILY_STRESS", "I'm anxious all the time.", 5),
    ("TODO_COMPLETED", "I finish all of my tasks.", 10),
    ("TODO_COMPLETED", "I finish about half of my to-do list.", 5),
    ("TODO_COMPLETED", "I get through most of my list.", 8),
    ("TODO_COMPLETED", "I complete 90% of my tasks.", 9),
    ("WEEKLY_MEDITATION", "I meditate every day.", 7),
    ("DONATION", "I volunteer every week.", 5),
]

NEGATIONS = [
    ("DONATION", "I never donate to charity.", 0),
    ("WEEKLY_MEDITATION", "I don't meditate.", 0),
    ("PLACES_VISITED", "I haven't visited anywhere new this year.", 0),
    ("DAILY_SHOUTING", "I never shout at anyone.", 0),
    ("CORE_CIRCLE", "I don't have any close friends.", 0),
    ("SUPPORTING_OTHERS", "I don't help anyone.", 0),
    ("TIME_FOR_PASSION", "I have no hobbies.", 0),
    ("SOCIAL_NETWORK", "I don't talk to anyone most days.", 0),
]

INTERFERENCE = [
    ("I sleep 7 hours and talk to 5 people a day.", {"SLEEP_HOURS": 7, "SOCIAL_NETWORK": 5}, []),
    ("I have two kids and sleep 6 hours a night.", {"SLEEP_HOURS": 6},
     ["CORE_CIRCLE", "SUPPORTING_OTHERS"]),
    ("I walk 8,000 steps and eat 3 portions of fruit a day.",
     {"DAILY_STEPS": 8, "FRUITS_VEGGIES": 3}, []),
    ("I work 10 hours a day and sleep 6 hours.", {"SLEEP_HOURS": 6}, ["FLOW"]),
    ("After 3 years at this job, I still love it.", {}, ["LIVE_VISION", "TIME_FOR_PASSION"]),
    ("I spend 2 hours a day gardening, which is my passion, and sleep 8 hours.",
     {"TIME_FOR_PASSION": 2, "SLEEP_HOURS": 8}, []),
    ("My 4 close friends and I walk about 6,000 steps together each day.",
     {"CORE_CIRCLE": 4, "DAILY_STEPS": 6}, []),
    ("I'm 35 and I meditate twice a week.", {"WEEKLY_MEDITATION": 2}, []),
]

NEUTRAL = [
    "Preheat the oven to 200 degrees. Mix two cups of flour with three eggs and bake for 25 minutes.",
    "Tomorrow will be cloudy with a high of 14 degrees and a 60 percent chance of rain in the afternoon.",
    "The headphones arrived in two days. Battery life is about 30 hours and the case feels solid.",
    "The meeting moved to Thursday at 3pm. Please bring the quarterly report and the budget figures.",
]

SAFETY_POSITIVE = [
    "Some days I think about killing myself.",
    "I keep thinking everyone would be better off without me.",
    "I've been self-harming again.",
    "I don't see the point in living anymore.",
    "I want to end it all.",
    "I feel like I want to die.",
]
SAFETY_NEGATIVE = [
    "This commute is killing me.",
    "I did a killer workout this morning.",
    "I'm dying to see that film.",
    "The project is in danger of slipping.",
    "I laughed so hard I nearly died.",
    "My phone battery died twice today.",
]


TOLERANCE = 0.5


def found(info):
    return info["value"] is not None and info["confidence"] != "low"


def single(extractor, text):
    return extractor.extract({PROMPTS[0]: text})


def check(label, passed, total):
    print(f"  {label:<44} {passed}/{total}  ({passed / total:.0%})")
    return passed, total


def reads_as(ex, item, text, expected, failures, tolerance=TOLERANCE):
    info = single(ex, text)["items"][item]
    good = found(info) and abs(info["value"] - expected) <= tolerance
    if not good:
        failures.append(f"{item}: \"{text}\" -> {info['value']} ({info['confidence']}), "
                        f"expected {expected}")
    return good


def count_interference(ex, cases, failures):
    passed = total = 0
    for text, expected, absent in cases:
        items = single(ex, text)["items"]
        for item, value in expected.items():
            total += 1
            good = found(items[item]) and abs(items[item]["value"] - value) <= TOLERANCE
            passed += good
            if not good:
                failures.append(f"\"{text}\" -> {item} = {items[item]['value']}, expected {value}")
        for item in absent:
            total += 1
            good = not found(items[item])
            passed += good
            if not good:
                failures.append(f"\"{text}\" -> {item} = {items[item]['value']} "
                                "leaked (should be empty)")
    return passed, total


def check_safety(ex, positive, negative, failures):
    flagged = {text: single(ex, text)["safety_concern"] for text in positive + negative}
    failures += [f"NOT flagged: \"{t}\"" for t in positive if not flagged[t]]
    failures += [f"wrongly flagged: \"{t}\"" for t in negative if flagged[t]]
    check("crisis wording flagged", sum(flagged[t] for t in positive), len(positive))
    check("everyday expressions not flagged", sum(not flagged[t] for t in negative), len(negative))


def run_templates(ex):
    print("== 1. Phrasing templates (each value stated one way) ==")
    failures, passed, total = [], 0, 0
    for item, (templates, values, scale) in TEMPLATES.items():
        results = [reads_as(ex, item, template.format(d=v, w=WORDS[v]),
                            scale(v) if scale else v, failures)
                   for template in templates for v in values]
        check(item, sum(results), len(results))
        passed += sum(results)
        total += len(results)
    phrases = [reads_as(ex, item, text, expected, failures) for item, text, expected in LEVEL_PHRASES]
    check("Phrases (stress, to-do, frequency words)", sum(phrases), len(phrases))
    check("TOTAL", passed + sum(phrases), total + len(phrases))
    return failures


def run_negations(ex):
    print("\n== 2. Negation ==")
    failures = []
    results = [reads_as(ex, item, text, expected, failures, tolerance=0)
               for item, text, expected in NEGATIONS]
    check("negated statements read as 0", sum(results), len(results))
    return failures


def run_interference(ex):
    print("\n== 3. Two facts in one sentence ==")
    failures = []
    check("correct item gets the number, no leaks", *count_interference(ex, INTERFERENCE, failures))
    return failures


def run_neutral(ex):
    print("\n== 4. Unrelated text (nothing should be extracted) ==")
    failures, passed = [], 0
    for text in NEUTRAL:
        items = single(ex, text)["items"]
        leaked = [f"{i}={v['value']:g}" for i, v in items.items() if found(v)]
        passed += not leaked
        if leaked:
            failures.append(f"\"{text[:60]}...\" -> {', '.join(leaked)}")
    check("unrelated paragraphs with nothing extracted", passed, len(NEUTRAL))
    return failures


def run_safety(ex):
    print("\n== 5. Safety flag ==")
    failures = []
    check_safety(ex, SAFETY_POSITIVE, SAFETY_NEGATIVE, failures)
    return failures


def swap_typo(text, rng):
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if candidates:
        i = rng.choice(candidates)
        w = words[i]
        j = rng.randrange(len(w) - 1)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def digits_to_words(text):
    return re.sub(r"\b(10|[1-9])\b(?![,.]\d)", lambda m: WORDS[int(m.group(1))], text)


def words_to_digits(text):
    lookup = {w: str(n) for n, w in WORDS.items()}
    return re.sub(r"\b(" + "|".join(lookup) + r")\b", lambda m: lookup[m.group(1).lower()], text,
                  flags=re.IGNORECASE)


TRANSFORMS = {
    "all lowercase": lambda t, rng: t.lower(),
    "no apostrophes": lambda t, rng: t.replace("'", ""),
    "digits -> words": lambda t, rng: digits_to_words(t),
    "words -> digits": lambda t, rng: words_to_digits(t),
    "filler sentences added": lambda t, rng: t + " The weather was nice. I had pasta for dinner.",
    "one typo per entry": swap_typo,
}


def signature(result):
    return {i: (round(v["value"], 2) if found(v) else None) for i, v in result["items"].items()}


def run_perturbations(ex):
    print("\n== 6. Small rewrites of the journal cases (answers should not change) ==")
    with open(os.path.join(HERE, "journal_cases.json")) as f:
        cases = json.load(f)["cases"]
    failures = []
    baselines = {c["name"]: signature(ex.extract(dict(zip(PROMPTS, c["entries"])))) for c in cases}
    for name, transform in TRANSFORMS.items():
        rng = random.Random(0)
        same = total = 0
        for c in cases:
            changed = [transform(e, rng) for e in c["entries"]]
            sig = signature(ex.extract(dict(zip(PROMPTS, changed))))
            for item, value in baselines[c["name"]].items():
                total += 1
                if sig[item] == value:
                    same += 1
                elif len(failures) < 40:
                    failures.append(f"[{name}] {c['name']} {item}: {value} -> {sig[item]}")
        check(name, same, total)
    return failures


def run_edges(ex):
    print("\n== 7. Edge cases and speed ==")
    notes = []
    for label, text in [("emoji", "I sleep 7 hours 😴 and walk 5,000 steps 🚶"),
                        ("huge number", "I sleep 1000 hours a night."),
                        ("non-English", "Duermo siete horas y camino cinco mil pasos."),
                        ("punctuation only", "...!!!???"),
                        ("one very long line", "I sleep 7 hours. " * 400)]:
        try:
            items = single(ex, text)["items"]
            got = {i: v["value"] for i, v in items.items() if found(v)}
            notes.append(f"  {label:<20} ok   -> {got if got else 'nothing extracted'}")
        except Exception as e:
            notes.append(f"  {label:<20} CRASH -> {type(e).__name__}: {e}")
    print("\n".join(notes))

    with open(os.path.join(HERE, "journal_cases.json")) as f:
        cases = json.load(f)["cases"]
    start = time.perf_counter()
    for c in cases:
        ex.extract(dict(zip(PROMPTS, c["entries"])))
    per_journal = (time.perf_counter() - start) / len(cases) * 1000
    long_journal = {p: " ".join(c["entries"][k] for c in cases) * 3 for k, p in enumerate(PROMPTS)}
    words = sum(len(v.split()) for v in long_journal.values())
    start = time.perf_counter()
    ex.extract(long_journal)
    long_ms = (time.perf_counter() - start) * 1000
    print(f"  speed: {per_journal:.0f} ms per typical journal; "
          f"{long_ms:.0f} ms for a {words:,}-word journal")
    return []


def run_unseen(ex):
    print("\n== 8. Unseen sentences (never used to tune the rules: the fair test) ==")
    with open(os.path.join(HERE, "nlp_unseen.json")) as f:
        data = json.load(f)
    failures = []
    values = [reads_as(ex, *case, failures) for case in data["values"]]
    check("values read correctly", sum(values), len(values))
    negations = [reads_as(ex, *case, failures) for case in data["negations"]]
    check("negations read as 0", sum(negations), len(negations))
    check("two facts in one sentence", *count_interference(ex, data["interference"], failures))
    check_safety(ex, data["safety_positive"], data["safety_negative"], failures)
    return failures


def main():
    scorer = Scorer()
    ex = make_extractor(scorer.items)
    ex.extract({PROMPTS[0]: "warm up"})

    failures = {}
    for name, fn in [("templates", run_templates), ("negation", run_negations),
                     ("interference", run_interference), ("neutral", run_neutral),
                     ("safety", run_safety), ("rewrites", run_perturbations), ("edges", run_edges),
                     ("unseen", run_unseen)]:
        failures[name] = fn(ex)

    print("\n== Failures ==")
    for name, items in failures.items():
        if items:
            print(f"\n[{name}]")
            for line in items:
                print(f"  {line}")


main()
