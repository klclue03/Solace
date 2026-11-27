import json
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "model"), ROOT]

DB = os.path.join(tempfile.mkdtemp(), "api_test.db")
os.environ["WELLBEING_DB"] = DB

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from scoring import Scorer  # noqa: E402

JOURNAL_ENTRIES = [
    "I lose track of time coding for about 5 hours a day. I get through most of my to-do list.",
    "Work is a bit stressful. I never shout at anyone.",
    "I volunteer every month. I have four close friends.",
    "I walked a bit and eat three servings of fruit a day.",
    "I talk to about 10 people a day and sleep 7 hours.",
]
CRISIS_ENTRY = "I don't see the point in living anymore."

results = []


def check(name, condition, detail=""):
    condition = bool(condition)
    results.append(condition)
    shown_detail = f"  ({detail})" if detail and not condition else ""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{shown_detail}")


def query(sql, *params):
    con = sqlite3.connect(DB)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def middle_answers(questions):
    return {q["id"]: (q["min"] + q["max"]) // 2 for q in questions}


def support_level(body):
    return body["support"] and body["support"]["level"]


def check_basics(client):
    print("== Basics ==")
    r = client.get("/health")
    check("GET /health", r.status_code == 200 and r.json()["status"] == "ok")
    version = r.json()["model_version"]

    r = client.get("/questions")
    body = r.json()
    questions = body["questions"]
    check("GET /questions returns 18 questions", r.status_code == 200 and len(questions) == 18)
    check("questions have ranges and labels",
          all({"id", "label", "question", "min", "max"} <= set(q) for q in questions))
    check("5 journal prompts and a privacy notice",
          len(body["journal_prompts"]) == 5 and body["privacy_notice"])
    check("OpenAPI contract and docs served",
          client.get("/openapi.json").status_code == 200 and client.get("/docs").status_code == 200)
    return questions, version


def check_survey(client, answers, version):
    print("\n== Survey responses ==")
    r = client.post("/responses", json={"user_id": "alice", "answers": answers})
    first = r.json()
    expected = Scorer().score(answers)["overall"]
    check("POST /responses returns 201", r.status_code == 201, r.text[:200])
    check("score matches the scoring model", first["overall"] == expected,
          f"{first['overall']} vs {expected}")
    check("saved with an id and timestamp", first["id"] is not None and first["created_at"])
    check("first response has no previous/changes",
          first["previous"] is None and first["changes"] is None)
    check("ordinary answers get no support message", first["support"] is None)
    check("model version reported", first["model_version"] == version)

    better = dict(answers, DAILY_STRESS=1)
    second = client.post("/responses", json={"user_id": "alice", "answers": better}).json()
    expected_change = second["overall"] - first["overall"]
    check("second response compares with the first",
          second["previous"]["id"] == first["id"]
          and abs(second["changes"]["overall"] - expected_change) < 0.11)

    preview = client.post("/responses",
                          json={"user_id": "alice", "answers": answers, "save": False}).json()
    check("save=false scores without saving", preview["id"] is None)

    history = client.get("/users/alice/responses").json()["responses"]
    check("history lists saved responses in order",
          [h["id"] for h in history] == [first["id"], second["id"]])
    check("unknown user has empty history",
          client.get("/users/nobody/responses").json()["responses"] == [])

    stressed = client.post("/responses",
                           json={"user_id": "bob", "answers": dict(answers, DAILY_STRESS=5)}).json()
    check("stress 5/5 returns the low_score support message", support_level(stressed) == "low_score")


def check_validation(client, answers):
    print("\n== Validation errors ==")
    missing = {k: v for k, v in answers.items() if k != "FLOW"}
    cases = [
        ("missing answer", {"user_id": "x", "answers": missing}),
        ("out-of-range answer", {"user_id": "x", "answers": dict(answers, DAILY_STRESS=9)}),
        ("non-numeric answer", {"user_id": "x", "answers": dict(answers, FLOW="lots")}),
        ("invalid user id", {"user_id": "has spaces", "answers": answers}),
        ("empty user id", {"user_id": "", "answers": answers}),
        ("journal source without extraction",
         {"user_id": "x", "source": "journal", "answers": answers}),
        ("unknown source", {"user_id": "x", "source": "email", "answers": answers}),
    ]
    for name, payload in cases:
        r = client.post("/responses", json=payload)
        check(f"{name} -> 422", r.status_code == 422, f"got {r.status_code}")
    saved = query("SELECT COUNT(*) FROM responses WHERE user_id = 'x'")[0][0]
    check("rejected requests saved nothing", saved == 0)


def check_journal(client, questions):
    print("\n== Journal ==")
    r = client.post("/journal/extract", json={"entries": JOURNAL_ENTRIES})
    extraction = r.json()
    items, found, unclear = extraction["items"], extraction["found"], extraction["unclear"]
    check("POST /journal/extract returns 200", r.status_code == 200, r.text[:200])
    check("found + unclear cover all 18 items",
          sorted(found + unclear) == sorted(q["id"] for q in questions))
    check("stated facts extracted", items["SLEEP_HOURS"]["value"] == 7 and "SLEEP_HOURS" in found)
    check("low-confidence guess is listed as unclear, not found",
          items["DAILY_STEPS"]["confidence"] == "low" and "DAILY_STEPS" in unclear)
    check("no safety concern for ordinary text",
          extraction["safety_concern"] is False and extraction["support"] is None)

    middle = middle_answers(questions)
    final = {i: items[i]["value"] if i in found else middle[i] for i in middle}
    final["DAILY_STRESS"] = 3
    journal_payload = {"user_id": "carol", "source": "journal", "answers": final}
    r = client.post("/responses", json=dict(journal_payload, journal=extraction))
    check("journal response saved (201)", r.status_code == 201, r.text[:200])
    check("journal results include the info support line", support_level(r.json()) == "info")

    source, stored = query("SELECT source, extraction FROM responses WHERE user_id = 'carol'")[0]
    details = json.loads(stored)["items"]
    check("stored as source=journal", source == "journal")
    check("origins recorded: journal / corrected / follow_up",
          details["SLEEP_HOURS"]["origin"] == "journal"
          and details["DAILY_STRESS"]["origin"] == "corrected"
          and details["DAILY_STEPS"]["origin"] == "follow_up")
    everything_stored = json.dumps(query("SELECT * FROM responses"))
    check("full journal answers never stored (only single-sentence quotes)",
          not any(e in everything_stored for e in JOURNAL_ENTRIES if e.count(".") > 1))

    crisis = client.post("/journal/extract", json={"entries": [CRISIS_ENTRY]}).json()
    check("crisis wording -> safety_concern and crisis support",
          crisis["safety_concern"] is True and support_level(crisis) == "crisis")

    too_many = client.post("/journal/extract", json={"entries": ["a"] * 6})
    empty = client.post("/journal/extract", json={"entries": ["", "  "]})
    too_long = client.post("/journal/extract", json={"entries": ["x" * 10_001]})
    check("more than 5 answers -> 422", too_many.status_code == 422)
    check("empty journal -> 422", empty.status_code == 422)
    check("answer over 10,000 characters -> 422", too_long.status_code == 422)

    incomplete = dict(extraction, items={"SLEEP_HOURS": items["SLEEP_HOURS"]})
    r = client.post("/responses", json=dict(journal_payload, journal=incomplete))
    check("incomplete journal extraction -> 422", r.status_code == 422)


def main():
    with TestClient(app) as client:
        questions, version = check_basics(client)
        answers = middle_answers(questions)
        check_survey(client, answers, version)
        check_validation(client, answers)
        check_journal(client, questions)
    print(f"\n{sum(results)}/{len(results)} API checks passed")


main()
