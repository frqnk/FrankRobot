import dotenv
import flask
import goose3
import io
import logging
import os
import queue
import requests
import spacy
import sys
import threading
import wikipediaapi
import wordcloud

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if not (TELEGRAM_BOT_TOKEN := os.getenv("TELEGRAM_BOT_TOKEN")):
    logging.critical("TELEGRAM_BOT_TOKEN is not set.")
    sys.exit(1)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = flask.Flask(__name__)
processing_queue = queue.Queue()
nlp = spacy.load("en_core_web_sm")
wiki = wikipediaapi.Wikipedia(
    user_agent="FrankRobot (frank.schlemmermeyer@fatec.sp.gov.br)"
)


@app.route("/", methods=["GET"])
def index():
    return "everything is awesome", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    if (
        (data := flask.request.get_json(silent=True)) is None
        or (message := data.get("message")) is None
        or (message_from := message.get("from")) is None
    ):
        logging.warning(f"Webhook data is incomplete or malformed: {data}\n")
        return "ignored", 400

    logging.info(
        f'{[            
            message_from.get("timestamp", ""),
            message_from.get("id", ""),
            message_from.get("first_name", ""),
            message_from.get("last_name", ""),
            message_text := message.get("text", ""),
        ]}'
    )

    processing_queue.put((message["chat"].get("id"), message_text))

    return "ok", 200


def worker():
    while True:
        message_chat_id, message_text = processing_queue.get()

        if message_text.startswith("/start"):
            sendMessage(
                message_chat_id,
                "Send /wordcloud or /wc followed by text, URL, or a Wikipedia article title to generate a word cloud.",
            )
            processing_queue.task_done()
            continue

        if message_text.startswith("/wordcloud") or message_text.startswith("/wc"):
            process_wordcloud(message_chat_id, message_text)
            processing_queue.task_done()
            continue

        processing_queue.task_done()


def process_wordcloud(message_chat_id, message_text):
    logging.info(
        f"[chat_id={message_chat_id}] Processing wordcloud for: {message_text}"
    )
    response_id = (
        sendMessage(message_chat_id, "Processing wordcloud...")
        .json()
        .get("result")
        .get("message_id")
    )
    try:
        with io.BytesIO() as image_buffer:
            wordcloud.WordCloud(width=1024, height=1024).generate(
                preprocessing(get_base_text(message_text))
            ).to_image().save(image_buffer, format="PNG")
            image_buffer.seek(0)
            editMessageMedia(message_chat_id, response_id, image_buffer)
        logging.info(f"[chat_id={message_chat_id}] Processing completed successfully")
    except ValueError as ve:
        logging.warning(f"[chat_id={message_chat_id}] Value error: {ve}")
        editMessageText(message_chat_id, response_id, f"Error: {ve}")
    except Exception as e:
        logging.exception(
            f"[chat_id={message_chat_id}] Unexpected error while generating word cloud: {e}"
        )
        editMessageText(
            message_chat_id,
            response_id,
            "Unexpected error while generating word cloud. Please try again later.",
        )


def get_base_text(message_text):
    if not message_text or not message_text.strip():
        raise ValueError("Text is empty.")

    if len(message_text) > 4096:
        raise ValueError("Text is too long (max 4096 characters).")

    if nlp(message_text.split()[0])[0].like_url:
        return goose3.Goose().extract(message_text).cleaned_text

    if (wiki_page := wiki.page(message_text[:256])).exists():
        return wiki_page.text

    if len(message_text.split()) < 7:
        raise ValueError("Text is too short (min 7 words).")

    return message_text


def preprocessing(text):
    return " ".join(
        [
            token.lemma_ if token.pos_ == "PROPN" else token.lemma_.lower()
            for token in nlp(text)
            if (
                token.is_alpha
                and token.pos_ not in {"PRON", "DET", "PART", "AUX"}
                and not token.is_stop
                and not token.is_punct
                and not token.like_url
                and not token.like_email
            )
        ]
    )


def sendMessage(chat_id, text):
    return requests.post(
        f"{TELEGRAM_API_URL}/sendMessage",
        data={"chat_id": f"{chat_id}", "text": f"{text}"},
    )


def editMessageText(chat_id, message_id, text):
    return requests.post(
        f"{TELEGRAM_API_URL}/editMessageText",
        data={
            "chat_id": f"{chat_id}",
            "message_id": f"{message_id}",
            "text": f"{text}",
        },
    )


def sendPhoto(chat_id, photo):
    return requests.post(
        f"{TELEGRAM_API_URL}/sendPhoto",
        files={"photo": photo},
        data={"chat_id": f"{chat_id}"},
    )


def editMessageMedia(chat_id, message_id, photo):
    return requests.post(
        f"{TELEGRAM_API_URL}/editMessageMedia",
        files={"photo": photo},
        data={
            "chat_id": f"{chat_id}",
            "message_id": f"{message_id}",
            "media": flask.json.dumps({"type": "photo", "media": "attach://photo"}),
        },
    )


def deleteMessage(chat_id, message_id):
    return requests.post(
        f"{TELEGRAM_API_URL}/deleteMessage",
        data={"chat_id": f"{chat_id}", "message_id": f"{message_id}"},
    )


if __name__ == "__main__":
    for _ in range(int(os.getenv("WORKERS", "1"))):
        threading.Thread(target=worker, daemon=True).start()
    app.run(port=int(os.getenv("PORT", "5000")))
