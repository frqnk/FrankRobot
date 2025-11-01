from typing import List

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from repositories.wiki_repository import WikiRepository
from services.knowledge_service import KnowledgeService
from services.text_processing_service import TextProcessingService


class QAService:
    def __init__(
        self,
        knowledge_service: KnowledgeService,
        wiki_repository: WikiRepository,
        text_processing_service: TextProcessingService,
        similarity_threshold: float = 0.05,
    ) -> None:
        self._knowledge_service = knowledge_service
        self._wiki_repository = wiki_repository
        self._text_processing_service = text_processing_service
        self._similarity_threshold = similarity_threshold

    def answer(self, user_text: str) -> str:
        base_sentences = self._knowledge_service.get_sentences()
        wiki_sentences = self._sentences_from_wikipedia(user_text)
        candidate_sentences = base_sentences + wiki_sentences
        processed_candidates = [
            self._text_processing_service.preprocess(sentence)
            for sentence in candidate_sentences
        ]
        processed_user_text = self._text_processing_service.preprocess(user_text)

        if not processed_candidates:
            return "Sorry, I have no answer for that."

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(processed_candidates + [processed_user_text])
        similarities = cosine_similarity(vectors[-1], vectors[:-1])

        best_index = similarities.argmax()
        best_score = similarities[0][best_index]

        if best_score < self._similarity_threshold:
            return "Sorry, I have no answer for that."

        if best_index < len(base_sentences):
            return base_sentences[best_index]

        return wiki_sentences[best_index - len(base_sentences)]

    def _sentences_from_wikipedia(self, text: str) -> List[str]:
        wiki_text = self._wiki_repository.get_page_text(text)
        if not wiki_text:
            return []
        return nltk.sent_tokenize(wiki_text)
