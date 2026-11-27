import re

PEOPLE = {"people", "person", "friend", "colleague", "neighbour", "neighbor", "kid", "child",
          "partner", "wife", "husband", "brother", "sister", "mum", "mom", "dad", "mother",
          "father", "son", "daughter", "parent", "family", "cousin", "aunt", "uncle", "boyfriend",
          "girlfriend", "flatmate", "roommate", "student", "pupil", "client", "patient", "junior",
          "teammate", "coworker", "co-worker", "resident", "mentee", "volunteer"}

SAFETY_PATTERNS = [re.compile(p) for p in [
    r"\bkill(ing)? myself\b",
    r"\bsuicid",
    r"\bself[- ]?harm",
    r"\b(hurt|hurting|harm|harming|cut|cutting) myself\b",
    r"\b(end|ending|take|taking) my (own )?life\b",
    r"\bend(ing)? it all\b",
    r"\b(dont|do not|no longer) want to (live|be alive|be here|exist|wake up)\b",
    r"\bwant(ed)? to die\b",
    r"\bwish (i|id) (was|were) dead\b",
    r"\bbetter off (dead|without me)\b",
    r"\b(no|dont see (the|any)|cant see (the|any)|there is no|theres no) point (in|of) "
    r"(living|life|going on|being alive|being here)\b",
    r"\b(nothing|no reason|no one) to live for\b",
    r"\bno reason to (live|keep going)\b",
    r"\bcant (go|carry) on (anymore|any more|like this)\b",
    r"\bdont want to (go on|be around)\b",
    r"\bnot safe at home\b",
    r"\b(im|i am) (in danger|not safe)\b",
    r"\bi feel unsafe\b",
    r"\boverdose\b",
    r"\bdisappear (forever|for good)\b",
]]

STRESS_WORDS = ["stressed", "stressful", "anxious", "worried", "tense", "on edge",
                "under pressure", "overwhelmed"]

STRESS_INTENSITY = [
    ("extremely", 5), ("constantly", 5), ("always", 5), ("not very", 1), ("not too", 1),
    ("very", 4), ("really", 4), ("super", 4), ("so", 4), ("pretty", 3), ("quite", 3),
    ("fairly", 3), ("a bit", 2), ("a little", 2), ("slightly", 2), ("somewhat", 2), ("not", 0),
]


def stress_levels():
    levels = [("very little stress", 1, "medium"), ("little stress", 1, "medium"),
              ("not much stress", 1, "medium"), ("no stress", 0, "medium"),
              ("stress-free", 0, "medium"), ("a lot of stress", 4, "medium"),
              ("stresses me out a lot", 4, "medium"), ("stress me out a lot", 4, "medium"),
              ("stresses me out", 3, "medium"), ("stress me out", 3, "medium")]
    for word in STRESS_WORDS:
        levels += [(f"{word} all the time", 5, "medium"), (f"{word} most of the time", 4, "medium")]
    for modifier, value in STRESS_INTENSITY:
        levels += [(f"{modifier} {word}", value, "medium") for word in STRESS_WORDS]
    levels += [("overwhelmed", 5, "medium"), ("on edge", 4, "medium"), ("very calm", 0, "medium"),
               ("calm", 1, "medium"), ("relaxed", 1, "medium"), ("stressed", 3, "low"),
               ("stressful", 3, "low"), ("stress", 3, "low"), ("anxious", 3, "low"),
               ("pressure", 3, "low"), ("worried", 3, "low"), ("tense", 3, "low")]
    return levels


def donation_scale(times_per_year):
    for limit, score in [(0, 0), (1, 1), (3, 2), (12, 3), (51, 4)]:
        if times_per_year <= limit:
            return score
    return 5


RULES = {
    "FLOW": {
        "keywords": ["absorbed", "absorb", "focus", "focused", "flow", "lose track of time",
                     "in the zone", "engrossed", "concentrate"],
        "units": ["hour", "minute"],
        "per": "day",
        "require_unit": True,
        "levels": [("lose track of time", 5, "low"), ("rarely focus", 1, "medium"),
                   ("can't focus", 0, "medium"), ("cannot focus", 0, "medium")],
    },
    "TIME_FOR_PASSION": {
        "keywords": ["passion", "passionate", "hobby", "hobbies", "love doing", "love", "enjoy",
                     "favourite", "favorite"],
        "units": ["hour"],
        "per": "day",
        "require_unit": True,
        "levels": [("no time for hobbies", 0, "medium"), ("no hobbies", 0, "medium"),
                   ("no time for", 0, "low")],
    },
    "ACHIEVEMENT": {
        "keywords": ["proud", "proudest", "achievement", "accomplish", "accomplishment", "achieve"],
        "units": ["thing", "achievement", "accomplishment", "moment"],
        "levels": [("nothing special", 0, "medium"), ("nothing to be proud", 0, "medium"),
                   ("proudest", 1, "medium")],
    },
    "TODO_COMPLETED": {
        "keywords": ["to-do", "todo", "to do list", "list", "task", "assignment", "homework", "chores"],
        "units": [],
        "percent": True,
        "levels": [("nearly everything", 9, "medium"), ("almost everything", 9, "medium"),
                   ("everything", 10, "medium"), ("all of", 10, "medium"), ("most of", 8, "medium"),
                   ("three quarters", 7.5, "medium"), ("two thirds", 6.7, "medium"),
                   ("a third", 3.3, "medium"), ("a quarter", 2.5, "medium"),
                   ("half", 5, "medium"), ("some of", 5, "medium"), ("a few things", 3, "medium"),
                   ("a few tasks", 3, "medium"), ("barely", 2, "medium"), ("little of", 2, "medium")],
    },
    "LIVE_VISION": {
        "keywords": ["plan", "vision", "goal", "future", "where i'm going", "where i am going",
                     "what i want", "career"],
        "units": ["year"],
        "require_unit": True,
        "levels": [("no idea", 0, "medium"), ("no plan", 0, "medium"),
                   ("not sure what", 1, "medium"), ("don't know what", 1, "medium")],
    },
    "DAILY_STRESS": {
        "keywords": ["stress", "stressed", "stressful", "anxious", "anxiety", "pressure",
                     "overwhelmed", "on edge", "calm", "relaxed", "tense", "worried"],
        "units": [],
        "levels": stress_levels(),
    },
    "DAILY_SHOUTING": {
        "keywords": ["shout", "yell", "snap", "sulk", "scream", "lose my temper", "my voice",
                     "argue", "argument", "angry"],
        "units": ["time"],
        "per": "week",
        "levels": [("every day", 7, "medium"), ("daily", 7, "medium"), ("often", 5, "low"),
                   ("a few times", 3, "medium"), ("a couple of times", 2, "medium"),
                   ("sometimes", 2, "low"), ("rarely", 1, "medium")],
    },
    "LOST_VACATION": {
        "keywords": ["vacation", "holiday", "annual leave", "days off", "time off", "pto",
                     "my leave", "leave days", "days of leave"],
        "units": ["day"],
        "per": "year",
        "require_unit": True,
        "levels": [("took all", 0, "medium"), ("take all", 0, "medium"), ("taken all", 0, "medium"),
                   ("used all", 0, "medium"), ("used up", 0, "medium")],
    },
    "SUPPORTING_OTHERS": {
        "keywords": ["help", "mentor", "support", "care for", "look after", "coach", "tutor",
                     "teach", "babysit"],
        "units": sorted(PEOPLE),
        "levels": [],
    },
    "DONATION": {
        "keywords": ["donate", "donation", "volunteer", "charity", "fundraise", "good cause"],
        "units": ["time"],
        "per": "year",
        "scale": donation_scale,
        "levels": [("every week", 5, "medium"), ("weekly", 5, "medium"), ("regularly", 4, "medium"),
                   ("every month", 3, "medium"), ("monthly", 3, "medium"),
                   ("sometimes", 2, "medium"), ("occasionally", 2, "medium"), ("once", 1, "medium")],
    },
    "PERSONAL_AWARDS": {
        "keywords": ["award", "recognition", "recognised", "recognized", "prize", "honour",
                     "honor", "promoted", "promotion"],
        "units": ["award", "recognition", "prize", "time"],
        "levels": [],
    },
    "CORE_CIRCLE": {
        "keywords": ["close", "trust", "best friend", "closest"],
        "units": ["friend", "people", "person"],
        "levels": [],
        "count_people": True,
    },
    "FRUITS_VEGGIES": {
        "keywords": ["fruit", "vegetable", "veg", "veggie", "salad", "takeaway", "junk food",
                     "fast food"],
        "units": ["serving", "portion", "piece", "fruit"],
        "per": "day",
        "levels": [("every meal", 4, "medium"), ("mostly takeaway", 1, "medium"),
                   ("junk food", 1, "medium"), ("fast food", 1, "medium"), ("takeaway", 1, "low")],
    },
    "DAILY_STEPS": {
        "keywords": ["step", "walk", "desk all day", "sit all day", "run", "hike"],
        "units": ["step"],
        "steps": True,
        "per": "day",
        "require_unit": True,
        "levels": [("desk all day", 2, "medium"), ("sit all day", 2, "medium"),
                   ("a lot", 8, "low"), ("a bit", 3, "low")],
    },
    "PLACES_VISITED": {
        "keywords": ["anywhere new", "new place", "visit", "travel", "trip", "town", "city",
                     "country", "been anywhere"],
        "units": ["place", "town", "city", "country", "trip"],
        "per": "year",
        "levels": [],
    },
    "WEEKLY_MEDITATION": {
        "keywords": ["meditate", "meditation", "reflect", "reflection", "mindful", "pray", "journal"],
        "units": ["time"],
        "per": "week",
        "levels": [("every morning", 7, "medium"), ("every evening", 7, "medium"),
                   ("every night", 7, "medium"), ("every day", 7, "medium"), ("daily", 7, "medium"),
                   ("most nights", 5, "medium"), ("most mornings", 5, "medium"),
                   ("most days", 5, "medium"), ("a few times a week", 3, "medium"),
                   ("twice a week", 2, "medium"), ("once a week", 1, "medium"),
                   ("every week", 1, "medium"), ("weekly", 1, "medium"),
                   ("not a chance", 0, "medium")],
    },
    "SOCIAL_NETWORK": {
        "keywords": ["talk to", "talk with", "chat", "interact", "speak to", "speak with",
                     "see people", "see loads", "see lots", "see anyone", "see anybody",
                     "see no one", "see nobody", "meet", "socialise", "socialize"],
        "unit_keywords": ["see", "meet", "deal", "speak"],
        "units": ["people", "person"],
        "per": "day",
        "levels": [("hardly see anyone", 1, "medium"), ("barely see anyone", 1, "medium"),
                   ("don't see anyone", 0, "medium"), ("no one", 0, "medium"),
                   ("nobody", 0, "medium"), ("loads of people", 8, "low"),
                   ("lots of people", 8, "low")],
    },
    "SLEEP_HOURS": {
        "keywords": ["sleep", "slept", "asleep"],
        "units": ["hour"],
        "per": "day",
        "require_unit": True,
        "levels": [],
    },
}
