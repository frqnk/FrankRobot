from goose3 import Goose


class ContentRepository:
    def __init__(self) -> None:
        self._goose = Goose()

    def extract_clean_text(self, url: str) -> str:
        return self._goose.extract(url).cleaned_text
