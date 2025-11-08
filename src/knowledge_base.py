import json
import logging
import numpy

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeEntry:
    question: str
    answer: str
    tags: Sequence[str]


class KnowledgeBase:
    def __init__(self, data_path: Path, tokenizer: Callable[[str], List[str]]) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._entries: List[KnowledgeEntry] = self._load_entries(data_path)
        self._tokenizer = tokenizer
        self._vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(self._corpus())

    def _load_entries(self, path: Path) -> List[KnowledgeEntry]:
        if not path.exists():
            self._logger.error("Knowledge base file not found at %s", path)
            raise FileNotFoundError(f"Knowledge base file not found at {path}.")
        self._logger.info("Loading knowledge base from %s", path)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        entries = [
            KnowledgeEntry(
                question=item["question"],
                answer=item["answer"],
                tags=item.get("tags", []),
            )
            for item in payload
        ]
        self._logger.info("Loaded %s knowledge entries", len(entries))
        return entries

    def _corpus(self) -> List[str]:
        return [
            f"{entry.question} {' '.join(entry.tags)} {entry.answer}"
            for entry in self._entries
        ]

    def search(self, text: str, top_k: int = 3) -> List[Tuple[KnowledgeEntry, float]]:
        if not text.strip():
            self._logger.warning("Empty search query received; returning no results.")
            return []

        self._logger.debug("Searching knowledge base with query='%s'", text)
        query_vec = self._vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self._matrix)[0]
        ranked_indices = numpy.argsort(similarities)[::-1][:top_k]

        results: List[Tuple[KnowledgeEntry, float]] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            results.append((self._entries[idx], score))
        self._logger.debug("Found %s results", len(results))
        return results
