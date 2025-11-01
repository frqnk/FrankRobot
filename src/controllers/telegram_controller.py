import logging
from typing import Any, Dict, Optional

from flask import Blueprint, request

from models.telegram_message import TelegramMessage
from services.message_service import MessageService


def create_telegram_blueprint(message_service: MessageService) -> Blueprint:
    blueprint = Blueprint("telegram", __name__)

    @blueprint.route("/", methods=["GET"])
    def index() -> tuple[str, int]:
        return "everything is awesome", 200

    @blueprint.route("/webhook", methods=["POST"])
    def webhook() -> tuple[str, int]:
        payload = request.get_json(silent=True)
        message = _extract_message(payload)
        if message is None:
            logging.warning("Webhook data is incomplete or malformed: %s", payload)
            return "ok", 200

        logging.info(
            "%s",
            [
                message.get("from", {}).get("timestamp", ""),
                message.get("from", {}).get("id", ""),
                message.get("from", {}).get("first_name", ""),
                message.get("from", {}).get("last_name", ""),
                message.get("text", ""),
            ],
        )

        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")
        if chat_id is None:
            logging.warning("Webhook message missing chat id: %s", message)
            return "ok", 200

        message_service.enqueue_message(
            TelegramMessage(chat_id=int(chat_id), text=text)
        )
        return "ok", 200

    return blueprint


def _extract_message(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    return message if isinstance(message, dict) else None
