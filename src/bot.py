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
    await update.message.chat.send_action(ChatAction.TYPING)

    response_text, _ = chatbot.reply(update.message.text)
    await update.message.reply_text(response_text)


def main() -> None:
    knowledge_base_path = Path(__file__).resolve().parent / "knowledge_base.json"
    chatbot = HybridChatbot(knowledge_base_path=knowledge_base_path)
    application = build_application(chatbot)
    application.run_webhook(
        listen="0.0.0.0",
        port=5000,
        url_path="webhook",
        webhook_url="https://server.canadacentral.cloudapp.azure.com",
    )


if __name__ == "__main__":
    main()
