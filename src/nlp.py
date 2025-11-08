import functools
import spacy

from typing import Dict, Iterable, List, Optional
from langdetect import DetectorFactory, LangDetectException, detect
from spacy.language import Language
from transformers import pipeline
from spacy.cli import download


DetectorFactory.seed = 42


class NLPEngine:
    _MULTILINGUAL_MODEL = "xx_sent_ud_sm"
    _PIPELINE_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(self, supported_languages: Dict[str, str] | None) -> None:
        self.supported_languages = supported_languages or {
            "en": "en_core_web_sm",
            "pt": "pt_core_news_sm",
        }
        self._loaded_models: Dict[str, Language] = {}
        self._sentiment_pipeline = pipeline(
            "text-classification", model=self._PIPELINE_MODEL
        )

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def _resolve_model_name(self, lang_code: str) -> str:
        base = lang_code.split("-")[0]
        return self.supported_languages.get(base, self._MULTILINGUAL_MODEL)

    @functools.lru_cache(maxsize=3)
    def _load_model(self, model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError:
            download(model_name)
            return spacy.load(model_name)

    def _get_nlp(self, lang_code: str) -> Language:
        model_name = self._resolve_model_name(lang_code)
        return self._load_model(model_name)

    def preprocess(self, text: str, lang_code: Optional[str] = None) -> str:
        lang = lang_code or self.detect_language(text)
        nlp = self._get_nlp(lang)
        doc = nlp(text)

        tokens: Iterable[str] = (
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_space and not token.is_stop
        )
        return " ".join(tokens)

    def tokenize(self, text: str, lang_code: Optional[str] = None) -> List[str]:
        preprocessed = self.preprocess(text, lang_code=lang_code)
        return [token for token in preprocessed.split() if token]

    def sentiment(
        self, text: str, lang_code: Optional[str] = None
    ) -> Dict[str, float | str]:
        lang = lang_code or self.detect_language(text)
        result = self._sentiment_pipeline(text, truncation=True)[0]
        score = float(result.get("score", 0.0))
        label = str(result.get("label", "neutral"))

        if label.lower().startswith("positive"):
            return {"label": "positive", "score": score, "language": lang}
        if label.lower().startswith("negative"):
            return {"label": "negative", "score": score, "language": lang}
        else:
            return {"label": "neutral", "score": score, "language": lang}