from os import getenv
import sys
import dotenv
import logging
import telegram_bot as bot

dotenv.load_dotenv()

if not (TELEGRAM_BOT_TOKEN := getenv("TELEGRAM_BOT_TOKEN")):
    logging.critical("TELEGRAM_BOT_TOKEN is not set.")
    sys.exit(1)

if __name__ == "__main__":
    bot.run(TELEGRAM_BOT_TOKEN)
