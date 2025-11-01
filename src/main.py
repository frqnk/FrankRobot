import logging
import sys
from os import getenv
from typing import Iterable

import dotenv
import nltk
from flask import Flask

from controllers.telegram_controller import create_telegram_blueprint
from repositories.content_repository import ContentRepository
from repositories.telegram_repository import TelegramRepository
from repositories.wiki_repository import WikiRepository
from services.knowledge_service import KnowledgeService
from services.message_service import MessageService
from services.qa_service import QAService
from services.text_processing_service import TextProcessingService
from services.wordcloud_service import WordCloudService

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

WELCOME_INPUT_WORDS = (
    "hey",
    "hello",
    "hi",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "yo",
    "sup",
    "hola",
    "oi",
    "olá",
    "salutations",
)
WELCOME_OUTPUT_WORDS = (
    "hey",
    "hello",
    "how you doing?",
    "welcome",
    "what's up?",
    "hi there!",
    "greetings!",
    "nice to see you!",
    "hello friend!",
    "good to see you!",
    "how can I help you today?",
)
WIKI_TOPICS: Iterable[str] = (
    "ChatGPT",
    "Natural language processing",
    "Machine learning",
    "Artificial intelligence",
    "Deep learning",
    "Transformer (machine learning model)",
    "Neural network",
    "Decision tree",
)


def ensure_nltk_models() -> None:
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def build_message_service(telegram_api_url: str) -> MessageService:
    text_processing_service = TextProcessingService()
    wiki_repository = WikiRepository(
        user_agent="FrankRobot (frank.schlemmermeyer@fatec.sp.gov.br)"
    )
    knowledge_service = KnowledgeService(wiki_repository, WIKI_TOPICS)
    qa_service = QAService(knowledge_service, wiki_repository, text_processing_service)
    content_repository = ContentRepository()
    wordcloud_service = WordCloudService(
        text_processing_service, wiki_repository, content_repository
    )
    telegram_repository = TelegramRepository(telegram_api_url)
    workers = int(getenv("WORKERS", "1"))
    return MessageService(
        telegram_repository,
        qa_service,
        wordcloud_service,
        WELCOME_INPUT_WORDS,
        WELCOME_OUTPUT_WORDS,
        workers,
    )


def create_app() -> Flask:
    ensure_nltk_models()
    telegram_bot_token = getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        logging.critical("TELEGRAM_BOT_TOKEN is not set.")
        raise SystemExit(1)

    telegram_api_url = f"https://api.telegram.org/bot{telegram_bot_token}"
    message_service = build_message_service(telegram_api_url)

    flask_app = Flask(__name__)
    flask_app.register_blueprint(create_telegram_blueprint(message_service))
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(port=int(getenv("PORT", "5000")))
