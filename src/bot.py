import tempfile
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
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    return application


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text(
        "Olá! Eu sou um chatbot. Envie uma mensagem de texto ou áudio e vou buscar a melhor resposta possível."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chatbot: HybridChatbot = context.application.chatbot
    await update.message.chat.send_action(ChatAction.TYPING)

    response_text, sentiment = chatbot.reply(update.message.text)
    await update.message.reply_text(response_text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chatbot: HybridChatbot = context.application.chatbot
    await update.message.chat.send_action(ChatAction.TYPING)

    voice_file = await update.message.voice.get_file()
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
        temp_path = Path(temp_audio.name)
        await voice_file.download_to_drive(temp_path)

    try:
        transcript = await chatbot.audio.voice_to_text(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if not transcript:
        await update.message.reply_text(
            "Não consegui entender o áudio. Pode tentar novamente com mais clareza?"
        )
        return

    await update.message.reply_text(f"Entendi: {transcript}")
    await update.message.chat.send_action(ChatAction.RECORD_VOICE)

    response_text, sentiment = chatbot.reply(transcript)
    await update.message.reply_text(response_text)

    audio_stream = await chatbot.audio.text_to_voice(
        response_text, sentiment.get("language", "en")
    )

    if audio_stream:
        await update.message.reply_voice(voice=audio_stream)


def main() -> None:
    knowledge_base_path = Path(__file__).resolve().parent / "knowledge_base.json"
    chatbot = HybridChatbot(
        knowledge_base_path=knowledge_base_path,
        huggingface_token=get_settings().huggingface_token,
    )
    application = build_application(chatbot)
    application.run_webhook(
        listen="0.0.0.0",
        port=5000,
        url_path="webhook",
        webhook_url="https://server.canadacentral.cloudapp.azure.com",
    )


if __name__ == "__main__":
    main()
