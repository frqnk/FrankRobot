from threading import Lock

import spacy


class TextProcessingService:
    def __init__(self, model_name: str = "en_core_web_lg") -> None:
        self._nlp = spacy.load(model_name)
        self._lock = Lock()

    def preprocess(self, text: str) -> str:
        with self._lock:
            doc = self._nlp(text)
        return " ".join(
            token.lemma_ if token.pos_ == "PROPN" else token.lemma_.lower()
            for token in doc
            if (
                token.is_alpha
                and token.pos_ not in {"PRON", "DET", "PART", "AUX"}
                and not token.is_stop
                and not token.is_punct
                and not token.like_url
                and not token.like_email
            )
        )

    def is_url(self, value: str) -> bool:
        with self._lock:
            doc = self._nlp(value)
        return bool(doc) and doc[0].like_url
