import logging
import random
from queue import Queue
from threading import Thread
from typing import Iterable, Optional

from models.telegram_message import TelegramMessage
from repositories.telegram_repository import TelegramRepository
from services.qa_service import QAService
from services.wordcloud_service import WordCloudService


class MessageService:
    def __init__(
        self,
        telegram_repository: TelegramRepository,
        qa_service: QAService,
        wordcloud_service: WordCloudService,
        welcome_inputs: Iterable[str],
        welcome_outputs: Iterable[str],
        workers: int = 1,
    ) -> None:
        self._telegram_repository = telegram_repository
        self._qa_service = qa_service
        self._wordcloud_service = wordcloud_service
        self._welcome_inputs = {word.lower() for word in welcome_inputs}
        self._welcome_outputs = list(welcome_outputs)
        self._queue: Queue[TelegramMessage] = Queue()
        self._start_workers(workers)

    def enqueue_message(self, message: TelegramMessage) -> None:
        self._queue.put(message)

    def _start_workers(self, workers: int) -> None:
        for _ in range(max(1, workers)):
            thread = Thread(target=self._worker, daemon=True)
            thread.start()

    def _worker(self) -> None:
        while True:
            message = self._queue.get()
            try:
                self._handle_message(message)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Unexpected error while processing message: %s", exc)
            finally:
                self._queue.task_done()

    def _handle_message(self, message: TelegramMessage) -> None:
        logging.info(
            "[chat_id=%s] Processing message: %s", message.chat_id, message.text
        )
        response_id = self._send_processing_message(message)
        if response_id is None:
            return

        if message.text.startswith("/start"):
            self._send_start_message(message.chat_id, response_id)
        elif message.text.startswith("/wordcloud") or message.text.startswith("/wc"):
            self._process_wordcloud_command(message, response_id)
        elif welcome := self._detect_welcome(message.text):
            self._telegram_repository.edit_message_text(
                message.chat_id, response_id, f"Chatbot: {welcome}"
            )
        else:
            answer = self._qa_service.answer(message.text)
            self._telegram_repository.edit_message_text(
                message.chat_id, response_id, f"Chatbot: {answer}"
            )

        logging.info("[chat_id=%s] Processing completed successfully", message.chat_id)

    def _send_processing_message(self, message: TelegramMessage) -> Optional[int]:
        response = self._telegram_repository.send_message(
            message.chat_id, "Processing..."
        )
        if not response:
            return None
        result = response.get("result")
        if not isinstance(result, dict):
            logging.warning("Unexpected Telegram API response: %s", response)
            return None
        message_id = result.get("message_id")
        if message_id is None:
            logging.warning("Telegram API response missing message_id: %s", response)
            return None
        try:
            return int(message_id)
        except (TypeError, ValueError):
            logging.warning("Invalid message_id received from Telegram: %s", response)
            return None

    def _send_start_message(self, chat_id: int, message_id: int) -> None:
        self._telegram_repository.edit_message_text(
            chat_id,
            message_id,
            (
                "You can make questions about artificial intelligence and correlated topics "
                "or use /wordcloud (or /wc) followed by text, URL, or a Wikipedia article "
                "title to generate a word cloud."
            ),
        )

    def _process_wordcloud_command(
        self, message: TelegramMessage, response_id: int
    ) -> None:
        parts = message.text.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        try:
            image_buffer = self._wordcloud_service.generate(argument)
            self._telegram_repository.edit_message_media(
                message.chat_id, response_id, image_buffer
            )
        except ValueError as error:
            logging.warning("[chat_id=%s] Value error: %s", message.chat_id, error)
            self._telegram_repository.edit_message_text(
                message.chat_id, response_id, f"Error: {error}"
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception(
                "[chat_id=%s] Unexpected error while generating word cloud: %s",
                message.chat_id,
                exc,
            )
            self._telegram_repository.edit_message_text(
                message.chat_id,
                response_id,
                "Unexpected error while generating word cloud. Please try again later.",
            )

    def _detect_welcome(self, text: str) -> Optional[str]:
        for word in text.split():
            if word.lower() in self._welcome_inputs:
                return self._choose_welcome_response()
        return None

    def _choose_welcome_response(self) -> Optional[str]:
        if not self._welcome_outputs:
            return None
        return random.choice(self._welcome_outputs)
