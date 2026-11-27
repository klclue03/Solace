import re

from scoring import QUESTIONS
from spacy_rules import PEOPLE, RULES, SAFETY_PATTERNS

SPACY_MODEL = "en_core_web_sm"

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "couple": 2, "few": 3, "several": 4, "dozen": 12, "half": 0.5,
    "once": 1, "twice": 2, "thrice": 3,
}
ARTICLES = {"a", "an"}
NEGATIONS = {"never", "no", "nothing", "none", "nowhere", "not", "n't"}
NUMBER_DEPS = {"nummod", "quantmod", "det", "amod", "advmod", "compound"}
CLAUSE_DEPS = {"ROOT", "conj", "ccomp", "advcl", "parataxis"}
PERCENT_WORDS = ("%", "percent", "per cent")

RATE_WORDS = {"a", "an", "per", "each", "every"}
PERIOD_WORDS = RATE_WORDS | {"this", "last", "the"}
PERIOD_DAYS = {"day": 1, "night": 1, "week": 7, "month": 365 / 12, "year": 365}
PERIOD_ADVERBS = {"daily": "day", "nightly": "day", "weekly": "week", "monthly": "month",
                  "yearly": "year", "annually": "year"}
PERIOD_WINDOW = (-2, 5)

SWAP_TYPO_MIN_LENGTH = 4
EDIT_TYPO_MIN_LENGTH = 7
CONFIDENT_NUMBER_DISTANCE = 3
SHORT_QUESTION_TOKENS = 4
MAX_EVIDENCE_CHARS = 120

MENTIONED = "mentioned"
RANK = {"high": 3, "medium": 2, "low": 1}


def parse_number(token):
    text = token.lower_.replace(",", "")
    if text in WORD_NUMBERS:
        return WORD_NUMBERS[text]
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(k)?", text)
    if match:
        return float(match.group(1)) * (1000 if match.group(2) else 1)
    return None


def number_with_conjuncts(token):
    value = parse_number(token)
    if value is None:
        return None
    extra_half = any(c.lower_ == "half" for c in token.conjuncts)
    alternatives = [v for v in (parse_number(c) for c in token.conjuncts if c.lower_ != "half")
                    if v is not None]
    value = (value + sum(alternatives)) / (1 + len(alternatives))
    return value + 0.5 if extra_half else value


def quantity_of(unit):
    numbers, articles, half = [], [], False
    for child in unit.children:
        if child.lower_ == "half":
            half = True
        if child.dep_ not in NUMBER_DEPS and child.dep_ != "predet":
            continue
        if child.lower_ in ARTICLES:
            articles.append(child)
            continue
        if child.lower_ == "half":
            continue
        value = number_with_conjuncts(child)
        if value is not None:
            numbers.append(value)

    if numbers:
        return sum(numbers) / len(numbers)
    if unit.dep_ == "pobj" and unit.head.lower_ == "of":
        amount = parse_number(unit.head.head)
        if amount is not None:
            return amount
    if half:
        return 0.5
    return 1.0 if articles else None


def clause_of(token):
    while not (token.dep_ == "ROOT" or (token.dep_ in CLAUSE_DEPS and token.pos_ in ("VERB", "AUX"))):
        token = token.head
    return token.i


def safety_concern(text):
    normalized = re.sub(r"\s+", " ", text.lower().replace("'", "").replace("’", ""))
    return any(p.search(normalized) for p in SAFETY_PATTERNS)


def contains_phrase(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)


def one_typo_apart(word, target):
    if word == target:
        return True
    if len(word) == len(target):
        diffs = [i for i, (a, b) in enumerate(zip(word, target)) if a != b]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1 and len(target) >= SWAP_TYPO_MIN_LENGTH:
            i = diffs[0]
            return word[i] == target[i + 1] and word[i + 1] == target[i]
        return len(diffs) == 1 and len(target) >= EDIT_TYPO_MIN_LENGTH
    if abs(len(word) - len(target)) != 1 or len(target) < EDIT_TYPO_MIN_LENGTH:
        return False
    shorter, longer = sorted((word, target), key=len)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


class Sentence:
    def __init__(self, span):
        self.span = span
        self.lower = span.text.lower()
        self.lemmas = [t.lemma_.lower() for t in span]
        roots = sum(t.dep_ == "ROOT" for t in span)
        self.clauses = [0] * len(span) if roots > 1 else [clause_of(t) for t in span]

    def same_clause(self, i, j):
        return self.clauses[i] == self.clauses[j]

    def matches(self, i, word):
        token = self.span[i]
        return (token.lower_ == word or self.lemmas[i] == word
                or (token.is_alpha and one_typo_apart(token.lower_, word)))

    def token_at(self, char_offset):
        absolute = self.span.start_char + char_offset
        for i, t in enumerate(self.span):
            if t.idx + len(t.text) > absolute:
                return i
        return len(self.span) - 1

    def keyword_position(self, keywords):
        positions = []
        for keyword in keywords:
            if " " in keyword or "-" in keyword:
                match = contains_phrase(self.lower, keyword)
                if match:
                    positions.append(self.token_at(match.start()))
            else:
                positions += [i for i in range(len(self.span)) if self.matches(i, keyword)]
        return min(positions) if positions else None

    def closest(self, candidates, anchor):
        ranked = [(not self.same_clause(i, anchor), abs(i - anchor), value, i)
                  for value, i in candidates if value is not None]
        if not ranked:
            return None
        other_clause, distance, value, index = min(ranked)
        return value, index, not other_clause, distance

    def unit_quantity(self, units, anchor):
        candidates = []
        for i, t in enumerate(self.span):
            if any(self.matches(i, u) for u in units):
                candidates.append((quantity_of(t), i))
            elif parse_number(t) is not None and any(
                    c.dep_ == "quantmod" and c.lemma_.lower() in units for c in t.children):
                candidates.append((parse_number(t), i))
        return self.closest(candidates, anchor)

    def rate_quantity(self, anchor):
        tokens = list(self.span)
        candidates = []
        for i, t in enumerate(tokens[:-2]):
            value = parse_number(t) if t.lower_ not in ARTICLES else None
            if (value is not None and tokens[i + 1].lower_ in RATE_WORDS
                    and self.lemmas[i + 2] in PERIOD_DAYS):
                candidates.append((value, i))
        return self.closest(candidates, anchor)

    def percentage(self, anchor):
        candidates = []
        for i, t in enumerate(self.span):
            if t.lower_ in PERCENT_WORDS:
                numbers = [parse_number(c) for c in t.children if parse_number(c) is not None]
                if not numbers and i > 0:
                    numbers = [parse_number(self.span[i - 1])]
                candidates += [(n, i) for n in numbers if n is not None]
        return self.closest(candidates, anchor)

    def nearest_number(self, anchor):
        candidates = [(parse_number(t), i) for i, t in enumerate(self.span)
                      if t.lower_ not in ARTICLES and (t.like_num or t.lower_ in WORD_NUMBERS)]
        return self.closest(candidates, anchor)

    def period_near(self, index):
        lo, hi = PERIOD_WINDOW
        found = []
        for i in range(max(0, index + lo), min(len(self.span), index + hi + 1)):
            t = self.span[i]
            if t.lower_ in PERIOD_ADVERBS:
                found.append((abs(i - index), PERIOD_ADVERBS[t.lower_]))
            elif self.lemmas[i] in PERIOD_DAYS and i > 0 and self.span[i - 1].lower_ in PERIOD_WORDS:
                found.append((abs(i - index), self.lemmas[i]))
        return min(found)[1] if found else None

    def level(self, levels):
        for phrase, value, confidence in levels:
            if contains_phrase(self.lower, phrase):
                return value, confidence
        return None

    def negated(self, anchor):
        return any((t.lower_ in NEGATIONS or t.dep_ == "neg") and self.same_clause(i, anchor)
                   for i, t in enumerate(self.span))

    def people_listed(self):
        heads = [t for t in self.span if t.lemma_.lower() in PEOPLE and t.dep_ != "conj"]
        return max((1 + sum(c.lemma_.lower() in PEOPLE for c in h.conjuncts) for h in heads),
                   default=None)

    def evidence(self):
        text = self.span.text.strip()
        if len(text) <= MAX_EVIDENCE_CHARS:
            return text
        return text[:MAX_EVIDENCE_CHARS - 3] + "..."


def per_period(value, sentence, index, rule):
    base = rule.get("per")
    if base is None:
        return value
    period = sentence.period_near(index)
    if period is not None:
        value = round(value * PERIOD_DAYS[base] / PERIOD_DAYS[period], 2)
    return rule["scale"](value) if "scale" in rule else value


def counted_value(sentence, rule, anchor):
    found = sentence.unit_quantity(set(rule["units"]), anchor)
    if found is None and rule.get("per"):
        found = sentence.rate_quantity(anchor)
    if found is None:
        return None
    value, index, same_clause, _ = found
    if rule.get("steps") and value >= 100:
        value /= 1000
    return per_period(value, sentence, index, rule), "high" if same_clause else "medium"


def read_item(sentence, rule):
    anchor = sentence.keyword_position(rule["keywords"])
    if anchor is None:
        unit_anchor = sentence.keyword_position(rule.get("unit_keywords", []))
        if unit_anchor is None:
            return None
        result = counted_value(sentence, rule, unit_anchor)
        return result if result and result[1] == "high" else None

    result = counted_value(sentence, rule, anchor)
    if result is not None:
        return result

    if rule.get("percent"):
        found = sentence.percentage(anchor)
        if found is not None:
            value, _, same_clause, _ = found
            return value / 10, "high" if same_clause else "medium"

    level = sentence.level(rule["levels"])
    if level and level[1] != "low":
        return level

    if sentence.negated(anchor):
        return 0.0, "medium"

    if rule.get("count_people"):
        people = sentence.people_listed()
        if people:
            return float(people), "medium"

    if not rule.get("require_unit"):
        found = sentence.nearest_number(anchor)
        if found is not None:
            number, index, same_clause, distance = found
            confident = same_clause and distance <= CONFIDENT_NUMBER_DISTANCE
            return per_period(number, sentence, index, rule), "medium" if confident else "low"

    return level or MENTIONED


def split_sentences(doc):
    spans = list(doc.sents)
    merged, i = [], 0
    while i < len(spans):
        span = spans[i]
        is_short_question = span.text.strip().endswith("?") and len(span) <= SHORT_QUESTION_TOKENS
        if is_short_question and i + 1 < len(spans):
            merged.append(doc[span.start:spans[i + 1].end])
            i += 2
        else:
            merged.append(span)
            i += 1
    return merged


class SpacyExtractor:
    def __init__(self, items, nlp=None):
        self.items = list(items)
        self._nlp = nlp

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(SPACY_MODEL)
        return self._nlp

    def extract(self, entries):
        from journal import JournalError
        texts = [a.strip() for a in entries.values() if a.strip()]
        if not texts:
            raise JournalError("The journal is empty.")

        sentences = [Sentence(s) for doc in self.nlp.pipe(texts) for s in split_sentences(doc)]
        items = {item: self.best_reading(item, sentences) for item in self.items}
        return {"items": items, "safety_concern": safety_concern(" ".join(texts))}

    def best_reading(self, item, sentences):
        _, lo, hi = QUESTIONS[item]
        best = {"value": None, "confidence": "low", "evidence": ""}
        for sentence in sentences:
            result = read_item(sentence, RULES[item])
            if result is None:
                continue
            if result == MENTIONED:
                if not best["evidence"]:
                    best["evidence"] = sentence.evidence()
                continue
            value, confidence = result
            if best["value"] is None or RANK[confidence] > RANK[best["confidence"]]:
                best = {"value": float(min(max(value, lo), hi)), "confidence": confidence,
                        "evidence": sentence.evidence()}
        if best["value"] is None:
            best["evidence"] = ""
        return best
