import re

from src.types import Prompt


def score(prompt: Prompt, response_text: str) -> float:
    answers = re.findall(r"(?m)^####[ \t]*([+-]?\d+)[ \t]*$", response_text)
    return float(bool(answers) and int(answers[-1]) == int(prompt.answer))
