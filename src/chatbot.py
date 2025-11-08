from pathlib import Path
from typing import Dict, Optional, Tuple

from .audio import AudioManager
from .knowledge_base import KnowledgeBase
from .nlp import NLPEngine


class HybridChatbot:
    def __init__(
        self, knowledge_base_path: Path, huggingface_token: Optional[str] = None
    ) -> None:
        self._nlp = NLPEngine()
        self._knowledge_base = KnowledgeBase(
            knowledge_base_path, tokenizer=self._nlp.tokenize
        )
        self._audio = AudioManager(huggingface_token=huggingface_token)

        self._empathetic_suffix: Dict[str, str] = {
            "en": "I hear you. Let's work through this together.",
            "pt": "Eu entendo. Vamos resolver isso juntos.",
        }
        self._celebratory_suffix: Dict[str, str] = {
            "en": "That's great! Here's something that might help even more.",
            "pt": "Que ótimo! Aqui está algo que pode ajudar ainda mais.",
        }
        self._neutral_suffix: Dict[str, str] = {
            "en": "Here's what I found.",
            "pt": "Olhe o que encontrei.",
        }

    @property
    def nlp(self) -> NLPEngine:
        return self._nlp

    @property
    def audio(self) -> AudioManager:
        return self._audio

    def reply(self, user_text: str) -> Tuple[str, Dict[str, float | str]]:
        language = self._nlp.detect_language(user_text)
        preprocessed = self._nlp.preprocess(user_text, lang_code=language)
        matches = self._knowledge_base.search(preprocessed, top_k=3)

        if matches:
            primary_answer = matches[0][0].answer
        else:
            primary_answer = self._fallback_answer(language)

        sentiment = self._nlp.sentiment(user_text, lang_code=language)
        adapted = self._adapt_response(primary_answer, sentiment)

        return adapted, sentiment

    def _fallback_answer(self, language: str) -> str:
        lang = language.split("-")[0]
        if lang == "pt":
            return "Ainda não sei responder isso, mas posso procurar mais informações se você quiser."
        return "I do not know that yet, but I can keep learning if you give me more context."

    def _adapt_response(
        self, base_response: str, sentiment: Dict[str, float | str]
    ) -> str:
        label = sentiment.get("label", "neutral")
        language = sentiment.get("language", "en").split("-")[0]

        if label == "negative":
            suffix = self._empathetic_suffix.get(
                language, self._empathetic_suffix["en"]
            )
            return f"{suffix} {base_response}"

        if label == "positive":
            suffix = self._celebratory_suffix.get(
                language, self._celebratory_suffix["en"]
            )
            return f"{suffix} {base_response}"

        suffix = self._neutral_suffix.get(language, self._neutral_suffix["en"])
        return f"{suffix} {base_response}"
