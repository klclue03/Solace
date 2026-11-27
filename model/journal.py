PROMPTS = [
    "Describe a recent typical day. What absorbed you, what are you working towards, "
    "and how much did you get done?",
    "What has been stressing you lately? How have you handled frustration or conflict? "
    "Have you been taking your time off?",
    "Who have you helped, given to, or felt close to recently? Have you been recognised for anything?",
    "How active have you been? What have you been eating, where have you been, "
    "and do you get time to reflect or meditate?",
    "How many people do you talk to on a typical day, and how have you been sleeping?",
]


class JournalError(Exception):
    pass


def make_extractor(items):
    from spacy_extractor import SpacyExtractor
    return SpacyExtractor(items)
