import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREFIX = os.path.join(HERE, "wellbeing_index")

QUESTIONS = {
    "FLOW":              ("In a typical day, how many hours are you fully absorbed ('in flow') in what you do?", 0, 10),
    "TIME_FOR_PASSION":  ("How many hours a day do you spend doing what you are passionate about?", 0, 10),
    "ACHIEVEMENT":       ("How many remarkable achievements are you proud of?", 0, 10),
    "TODO_COMPLETED":    ("How well do you complete your weekly to-do list? (0 = not at all, 10 = fully)", 0, 10),
    "LIVE_VISION":       ("For how many years ahead is your life vision very clear?", 0, 10),
    "DAILY_STRESS":      ("How much stress do you typically feel each day? (0 = none, 5 = a lot)", 0, 5),
    "DAILY_SHOUTING":    ("How often do you shout at or sulk with someone in a typical week?", 0, 10),
    "LOST_VACATION":     ("How many days of vacation do you typically lose (leave unused) each year?", 0, 10),
    "SUPPORTING_OTHERS": ("How many people do you help to achieve a better life?", 0, 10),
    "DONATION":          ("How many times do you donate time or money to good causes? (0-5)", 0, 5),
    "PERSONAL_AWARDS":   ("How many recognitions or awards have you received?", 0, 10),
    "CORE_CIRCLE":       ("How many people are very close to you?", 0, 10),
    "FRUITS_VEGGIES":    ("How many servings of fruit and vegetables do you eat a day?", 0, 5),
    "DAILY_STEPS":       ("How many thousand steps do you walk a day?", 1, 10),
    "PLACES_VISITED":    ("How many new places do you visit in a year?", 0, 10),
    "WEEKLY_MEDITATION": ("How many times a week do you get time to reflect or meditate?", 0, 10),
    "SOCIAL_NETWORK":    ("How many people do you interact with on a typical day?", 0, 10),
    "SLEEP_HOURS":       ("How many hours do you sleep on a typical night?", 1, 10),
}


def approx_percentile(z):
    return 50 * (1 + math.erf(z / math.sqrt(2)))


class Scorer:
    def __init__(self, prefix=DEFAULT_PREFIX):
        with open(prefix + "_preprocess.json", "rb") as f:
            raw = f.read()
        self.version = hashlib.sha256(raw).hexdigest()[:12]
        p = json.loads(raw)

        import torch
        import encoder
        self._check_compatible(p, encoder)

        GroupedAE, _ = encoder.build_models()
        self.model = GroupedAE()
        self.model.load_state_dict(torch.load(prefix + ".pt"))
        self.model.eval()
        self._torch = torch

        self.features = p["features"]
        self.dims = p["dims"]
        self.neg = np.array([f in p["negative"] for f in self.features])
        self.mean = np.array(p["scaler_mean"])
        self.scale = np.array(p["scaler_scale"])
        self.signs = np.array(p["signs"])
        self.sub_mean = np.array(p["sub_mean"])
        self.sub_std = np.array(p["sub_std"])
        self.grid = np.array(p["percentile_grid"])
        self.dim_weights = np.array(p.get("dim_weights", [1.0] * len(self.dims)))

        sleep = p["sleep"]
        self.sleep_item = sleep["feature"]
        self.sleep_target = sleep["target_hours"]
        self.rest_mean = sleep["rest_mean"]
        self.rest_std = sleep["rest_std"]
        self.rest_weight = sleep["weight"]

    @staticmethod
    def _check_compatible(p, encoder):
        if p["features"] != encoder.FEATURES or p["dims"] != encoder.DIMS:
            raise RuntimeError("Saved model does not match GROUPS in encoder.py. "
                               "Re-run encoder.py to retrain before scoring.")
        if "signs" not in p:
            raise RuntimeError("Saved model has no 'signs'. Re-run encoder.py to retrain.")
        missing = [c for c in p["features"] + [p["sleep"]["feature"]] if c not in QUESTIONS]
        if missing:
            raise RuntimeError(f"No question text in scoring.QUESTIONS for: {missing}")

    @property
    def items(self):
        return self.features + [self.sleep_item]

    def validate(self, item, value):
        _, lo, hi = QUESTIONS[item]
        v = float(value)
        if not lo <= v <= hi:
            raise ValueError(f"{item} must be between {lo} and {hi}, got {value}")
        return v

    def _subscores(self, answers):
        x = np.array([answers[f] for f in self.features])
        x[self.neg] = -x[self.neg]
        z = (x - self.mean) / self.scale
        with self._torch.no_grad():
            sub = self.model.encode(self._torch.tensor(z[None], dtype=self._torch.float32))
        sub = sub.numpy()[0] * self.signs
        return (sub - self.sub_mean) / self.sub_std

    def _rest(self, answers):
        rest = -abs(answers[self.sleep_item] - self.sleep_target)
        return (rest - self.rest_mean) / self.rest_std

    def score(self, answers):
        missing = [i for i in self.items if i not in answers]
        if missing:
            raise ValueError(f"Missing answers for: {missing}")
        answers = {i: self.validate(i, answers[i]) for i in self.items}

        zsub = self._subscores(answers)
        zrest = self._rest(answers)

        w = np.r_[self.dim_weights, self.rest_weight]
        raw = float(np.r_[zsub, zrest] @ w / w.sum())
        overall = float(np.interp(raw, self.grid, np.arange(101)))

        return {
            "overall": round(overall, 1),
            "subscores": {d: round(approx_percentile(v), 1) for d, v in zip(self.dims, zsub)},
            "rest": round(approx_percentile(zrest), 1),
            "answers": answers,
            "model_version": self.version,
        }
