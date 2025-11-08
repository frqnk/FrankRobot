import logging
from pathlib import Path
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .chatbot import HybridChatbot
from .config import get_settings


logger = logging.getLogger(__name__)


def build_application(chatbot: HybridChatbot) -> Application:
    application = ApplicationBuilder().token(get_settings().telegram_token).build()
    application.chatbot = chatbot

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    return application


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text(
        "Olá! Eu sou um chatbot. Envie uma mensagem de texto e vou buscar a melhor resposta possível."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chatbot: HybridChatbot = context.application.chatbot
    logger.debug(
        "Handling text update from user_id=%s",
        update.effective_user.id if update.effective_user else "unknown",
    )
    await update.message.chat.send_action(ChatAction.TYPING)

    response_text, _ = chatbot.reply(update.message.text)
    await update.message.reply_text(response_text)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    logger.info("Initializing FrankRobot service")
    knowledge_base_path = Path(__file__).resolve().parent / "knowledge_base.json"
    logger.info("Using knowledge base at %s", knowledge_base_path)

    chatbot = HybridChatbot(knowledge_base_path=knowledge_base_path)
    application = build_application(chatbot)

    logger.info("Starting webhook listener")
    try:
        application.run_webhook(
            listen="0.0.0.0",
            port=5000,
            url_path="webhook",
            webhook_url="https://server.canadacentral.cloudapp.azure.com",
        )
    except Exception:  # pragma: no cover - crash diagnostics for service start
        logger.exception("FrankRobot service failed to start")
        raise


if __name__ == "__main__":
    main()
