import os

from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    telegram_token: str


def get_settings() -> Settings:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is missing.")

    return Settings(token)
