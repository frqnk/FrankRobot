from typing import Iterable, List

import nltk

from repositories.wiki_repository import WikiRepository


class KnowledgeService:
    def __init__(self, wiki_repository: WikiRepository, topics: Iterable[str]) -> None:
        self._wiki_repository = wiki_repository
        self._topics = list(topics)
        self._sentences = self._build_corpus()

    def get_sentences(self) -> List[str]:
        return self._sentences

    def _build_corpus(self) -> List[str]:
        corpus: List[str] = []
        for topic in self._topics:
            text = self._wiki_repository.get_page_text(topic)
            if text:
                corpus.extend(nltk.sent_tokenize(text))
        return corpus
