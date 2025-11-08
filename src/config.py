import os

from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    telegram_token: str
    huggingface_token: str | None


def get_settings() -> Settings:
    load_dotenv()

    return Settings(
        os.getenv("TELEGRAM_BOT_TOKEN"),
        os.getenv("HUGGINGFACE_API_TOKEN"),
    )
