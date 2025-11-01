from typing import Optional

from wikipediaapi import Wikipedia, WikipediaPage


class WikiRepository:
    def __init__(self, user_agent: str) -> None:
        self._wiki = Wikipedia(user_agent=user_agent)

    def get_page(self, title: str) -> WikipediaPage:
        return self._wiki.page(title[:256])

    def get_page_text(self, title: str) -> Optional[str]:
        page = self.get_page(title)
        return page.text if page.exists() else None
