import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextProcessor:
    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.stop_words = self._load_stopwords()

    def _load_stopwords(self) -> set:
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))

        # Keep important negation words
        stop_words = stop_words - {"no", "not", "nor"}
        return stop_words

    def clean_text(self, text: str) -> str:
        if text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens: List[str] = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.stemmer.stem(word) for word in tokens]

        return " ".join(tokens)
