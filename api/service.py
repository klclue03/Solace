import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
from journal import PROMPTS, JournalError, make_extractor  # noqa: E402
from scoring import DEFAULT_PREFIX, QUESTIONS, Scorer  # noqa: E402

DEFAULT_DB = os.path.join(ROOT, "data", "responses.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    answers       TEXT NOT NULL,
    subscores     TEXT NOT NULL,
    rest          REAL NOT NULL,
    overall       REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT 'survey',
    extraction    TEXT
);
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses (user_id, created_at);
"""

ADDED_COLUMNS = {
    "source": "TEXT NOT NULL DEFAULT 'survey'",
    "extraction": "TEXT",
}

PRIVACY_NOTICE = (
    "Your journal is analysed by this service and is never sent to anyone else. "
    "Your full text is not saved: only the answers we extract and short quotes "
    "supporting them are stored.")

HELPLINE = "https://findahelpline.com (in the US, call or text 988)."
NOT_ALONE = ("You don't have to face it alone. If you are in immediate danger, please call your "
             "local emergency number. You can find free, confidential support in your country at "
             + HELPLINE)

SUPPORT_MESSAGES = {
    "crisis": ("It sounds like you may be going through something really difficult right now. "
               + NOT_ALONE),
    "low_score": "Your answers suggest things may be really hard for you at the moment. " + NOT_ALONE,
    "info": "If things feel hard, free and confidential support is available at " + HELPLINE,
}

MAX_STRESS = 5
LOW_OVERALL = 10
SOURCES = ("survey", "journal")


def label(item):
    return item.replace("_", " ").capitalize()


def support(level):
    return {"level": level, "message": SUPPORT_MESSAGES[level]} if level else None


def is_found(info):
    return info["value"] is not None and info["confidence"] != "low"


def changes_since(previous, result):
    if not previous:
        return None
    return {
        "overall": round(result["overall"] - previous["overall"], 1),
        "subscores": {d: round(v - previous["subscores"][d], 1)
                      for d, v in result["subscores"].items() if d in previous["subscores"]},
        "rest": round(result["rest"] - previous["rest"], 1),
    }


class WellbeingService:
    def __init__(self, db_path=DEFAULT_DB, model_prefix=DEFAULT_PREFIX):
        self.db_path = db_path
        self.model_prefix = model_prefix
        self._scorer = None
        self._extractor = None
        with self._db():
            pass

    @property
    def scorer(self):
        if self._scorer is None:
            self._scorer = Scorer(self.model_prefix)
        return self._scorer

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = make_extractor(self.items)
        return self._extractor

    @property
    def items(self):
        return self.scorer.items

    @property
    def model_version(self):
        return self.scorer.version

    def warm_up(self):
        self.extractor.extract({PROMPTS[0]: "I sleep 7 hours."})

    def questions(self):
        return [{"id": i, "label": label(i), "question": QUESTIONS[i][0],
                 "min": QUESTIONS[i][1], "max": QUESTIONS[i][2]} for i in self.items]

    def validate(self, item, value):
        return self.scorer.validate(item, value)

    def score(self, answers):
        return self.scorer.score(answers)

    def extract_journal(self, entries):
        if isinstance(entries, (list, tuple)):
            if len(entries) > len(PROMPTS):
                raise JournalError(f"Expected at most {len(PROMPTS)} journal answers.")
            entries = dict(zip(PROMPTS, entries))
        extraction = self.extractor.extract(entries)
        items = extraction["items"]
        return {
            "items": items,
            "found": [i for i in self.items if is_found(items[i])],
            "unclear": [i for i in self.items if not is_found(items[i])],
            "safety_concern": extraction["safety_concern"],
            "support": support("crisis") if extraction["safety_concern"] else None,
        }

    def journal_details(self, extraction, answers):
        details = {}
        for item in self.items:
            info = extraction["items"][item]
            if not is_found(info):
                origin = "follow_up"
            elif abs(float(answers[item]) - info["value"]) < 1e-9:
                origin = "journal"
            else:
                origin = "corrected"
            details[item] = {"origin": origin, "confidence": info["confidence"],
                             "evidence": info["evidence"]}
        return {"items": details, "safety_concern": bool(extraction["safety_concern"])}

    def result_support(self, result, source, safety_concern=False):
        high_stress = result["answers"].get("DAILY_STRESS", 0) >= MAX_STRESS
        if (high_stress or result["overall"] < LOW_OVERALL) and not safety_concern:
            return support("low_score")
        if source == "journal" or safety_concern:
            return support("info")
        return None

    def submit(self, user_id, answers, source="survey", extraction=None, save=True):
        result = self.score(answers)
        previous = self.last_result(user_id)
        safety_concern = bool(extraction and extraction.get("safety_concern"))
        details = self.journal_details(extraction, result["answers"]) if extraction else None

        response_id = created_at = None
        if save:
            response_id, created_at = self.save(user_id, result, source, details)

        return {
            "id": response_id,
            "user_id": user_id,
            "created_at": created_at,
            "source": source,
            "model_version": result["model_version"],
            "overall": result["overall"],
            "subscores": result["subscores"],
            "rest": result["rest"],
            "answers": result["answers"],
            "previous": previous,
            "changes": changes_since(previous, result),
            "support": self.result_support(result, source, safety_concern),
        }

    @contextmanager
    def _db(self):
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            existing = {row[1] for row in con.execute("PRAGMA table_info(responses)")}
            for column, definition in ADDED_COLUMNS.items():
                if column not in existing:
                    con.execute(f"ALTER TABLE responses ADD COLUMN {column} {definition}")
            yield con
            con.commit()
        finally:
            con.close()

    def save(self, user_id, result, source, extraction=None):
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._db() as con:
            cursor = con.execute(
                "INSERT INTO responses (user_id, created_at, model_version, answers, "
                "subscores, rest, overall, source, extraction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, created_at, result["model_version"], json.dumps(result["answers"]),
                 json.dumps(result["subscores"]), result["rest"], result["overall"], source,
                 json.dumps(extraction) if extraction else None))
        return cursor.lastrowid, created_at

    def history(self, user_id):
        with self._db() as con:
            rows = con.execute(
                "SELECT id, created_at, source, model_version, overall, subscores, rest "
                "FROM responses WHERE user_id = ? ORDER BY created_at, id", (user_id,)).fetchall()
        return [{"id": rid, "created_at": at, "source": source, "model_version": version,
                 "overall": overall, "subscores": json.loads(subscores), "rest": rest}
                for rid, at, source, version, overall, subscores, rest in rows]

    def last_result(self, user_id):
        entries = self.history(user_id)
        return entries[-1] if entries else None
