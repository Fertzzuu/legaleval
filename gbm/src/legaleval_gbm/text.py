import re

import pymorphy3

_WORD_RE = re.compile(r"\w+")


class Lemmatizer:
    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        self._forms: dict[str, str] = {}

    def lemma(self, word: str) -> str:
        form = self._forms.get(word)
        if form is None:
            form = self._morph.parse(word)[0].normal_form
            self._forms[word] = form
        return form

    def lemmas(self, text: str) -> list[str]:
        return [self.lemma(w) for w in _WORD_RE.findall(text.lower())]


def is_number(token: str) -> bool:
    return token.isdigit()


def trigrams(tokens: list[str]) -> set[tuple[str, str, str]]:
    return set(zip(tokens, tokens[1:], tokens[2:], strict=False))


def windows(tokens: list[str], size: int, stride: int) -> list[list[str]]:
    if len(tokens) <= size:
        return [tokens]
    starts = range(0, len(tokens) - size + stride, stride)
    return [tokens[s : s + size] for s in starts]
