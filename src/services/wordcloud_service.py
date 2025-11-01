from io import BytesIO

from wordcloud import WordCloud

from repositories.content_repository import ContentRepository
from repositories.wiki_repository import WikiRepository
from services.text_processing_service import TextProcessingService


class WordCloudService:
    def __init__(
        self,
        text_processing_service: TextProcessingService,
        wiki_repository: WikiRepository,
        content_repository: ContentRepository,
        min_words: int = 7,
        max_length: int = 4096,
    ) -> None:
        self._text_processing_service = text_processing_service
        self._wiki_repository = wiki_repository
        self._content_repository = content_repository
        self._min_words = min_words
        self._max_length = max_length

    def generate(self, message_text: str) -> BytesIO:
        base_text = self._resolve_base_text(message_text)
        processed_text = self._text_processing_service.preprocess(base_text)
        buffer = BytesIO()
        WordCloud(width=1024, height=1024).generate(processed_text).to_image().save(
            buffer, format="PNG"
        )
        buffer.seek(0)
        return buffer

    def _resolve_base_text(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Text is empty.")

        if len(text) > self._max_length:
            raise ValueError(f"Text is too long (max {self._max_length} characters).")

        first_token = text.split()[0]
        if self._text_processing_service.is_url(first_token):
            return self._content_repository.extract_clean_text(text)

        wiki_text = self._wiki_repository.get_page_text(text)
        if wiki_text:
            return wiki_text

        if len(text.split()) < self._min_words:
            raise ValueError(
                "Wikipedia title did not match or input text is too short for a wordcloud (min 7 words)."
            )

        return text
