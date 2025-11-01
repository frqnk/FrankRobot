import json
import logging
from typing import IO, Any, Dict, Optional

import requests


class TelegramRepository:
    def __init__(self, api_url: str) -> None:
        self._api_url = api_url

    def send_message(self, chat_id: int, text: str) -> Optional[Dict[str, Any]]:
        response = self._post(
            "sendMessage", data={"chat_id": str(chat_id), "text": text}
        )
        return response.json() if response is not None else None

    def edit_message_text(
        self, chat_id: int, message_id: int, text: str
    ) -> Optional[Dict[str, Any]]:
        response = self._post(
            "editMessageText",
            data={
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "text": text,
            },
        )
        return response.json() if response is not None else None

    def edit_message_media(
        self, chat_id: int, message_id: int, photo: IO[bytes]
    ) -> Optional[Dict[str, Any]]:
        response = self._post(
            "editMessageMedia",
            data={
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "media": json.dumps({"type": "photo", "media": "attach://photo"}),
            },
            files={"photo": photo},
        )
        return response.json() if response is not None else None

    def delete_message(self, chat_id: int, message_id: int) -> Optional[Dict[str, Any]]:
        response = self._post(
            "deleteMessage",
            data={"chat_id": str(chat_id), "message_id": str(message_id)},
        )
        return response.json() if response is not None else None

    def _post(
        self,
        endpoint: str,
        data: Dict[str, str],
        files: Optional[Dict[str, IO[bytes]]] = None,
    ) -> Optional[requests.Response]:
        try:
            response = requests.post(
                f"{self._api_url}/{endpoint}", data=data, files=files
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logging.exception("Telegram API request failed: %s", exc)
            return None
