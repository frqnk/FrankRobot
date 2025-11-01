from dataclasses import dataclass


@dataclass
class TelegramMessage:
    chat_id: int
    text: str
