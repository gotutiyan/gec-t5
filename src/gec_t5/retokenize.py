import re

retokenization_rules = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" - ", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]

def retokenize(sent):
    for rule in retokenization_rules:
        sent = re.sub(rule[0], rule[1], sent)
    return sent.strip()