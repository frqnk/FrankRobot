import random
import dotenv
from flask import Flask, request, json
from goose3 import Goose
from io import BytesIO
import logging
from os import getenv
from queue import Queue
import requests
import spacy
import sys
from threading import Thread
from wikipediaapi import Wikipedia
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if not (TELEGRAM_BOT_TOKEN := getenv("TELEGRAM_BOT_TOKEN")):
    logging.critical("TELEGRAM_BOT_TOKEN is not set.")
    sys.exit(1)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = Flask(__name__)
processing_queue = Queue()
nlp = spacy.load("en_core_web_sm")
wiki = Wikipedia(user_agent="FrankRobot (frank.schlemmermeyer@fatec.sp.gov.br)")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

sentences = []
wiki_topics = [
    "ChatGPT",
    "Natural language processing",
    "Machine learning",
    "Artificial intelligence",
    "Deep learning",
    "Transformer (machine learning model)",
    "Neural network",
    "Decision tree",
]
for topic in wiki_topics:
    sentences.extend(
        [sentence for sentence in nltk.sent_tokenize(wiki.page(topic).text)]
    )

welcome_words_input = [
    "hey",
    "hello",
    "hi",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "yo",
    "sup",
    "hola",
    "salutations",
]
welcome_words_output = [
    "hey",
    "hello",
    "how you doing?",
    "welcome",
    "what's up?",
    "hi there!",
    "greetings!",
    "nice to see you!",
    "hello friend!",
    "good to see you!",
    "how can I help you today?",
]


@app.route("/", methods=["GET"])
def index():
    return "everything is awesome", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    if (
        (data := request.get_json(silent=True)) is None
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
        elif message_text.startswith("/wordcloud") or message_text.startswith("/wc"):
            process_wordcloud(message_chat_id, message_text)
        elif welcome := welcome_message(message_text):
            sendMessage(message_chat_id, f"Chatbot: {welcome}")
        else:
            sendMessage(
                message_chat_id,
                f"Chatbot: {answer(message_text, sentences)}",
            )

        processing_queue.task_done()


def welcome_message(text):
    for word in text.split():
        if word.lower() in welcome_words_input:
            return random.choice(welcome_words_output)
    return None


def answer(user_text, sentences, threshold=0.05):
    cleaned_user_text = preprocessing(user_text)
    cleaned_sentences = [preprocessing(s) for s in sentences]
    cleaned_sentences.append(cleaned_user_text)
    x_sentences = TfidfVectorizer().fit_transform(cleaned_sentences)
    similarity = cosine_similarity(x_sentences[-1], x_sentences)
    sentence_index = similarity.argsort()[0][-2]
    if similarity[0][sentence_index] < threshold:
        return "Sorry, I have no answer for that."
    else:
        return sentences[sentence_index]


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
        with BytesIO() as image_buffer:
            WordCloud(width=1024, height=1024).generate(
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
        return Goose().extract(message_text).cleaned_text

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
            "media": json.dumps({"type": "photo", "media": "attach://photo"}),
        },
    )


def deleteMessage(chat_id, message_id):
    return requests.post(
        f"{TELEGRAM_API_URL}/deleteMessage",
        data={"chat_id": f"{chat_id}", "message_id": f"{message_id}"},
    )


if __name__ == "__main__":
    for _ in range(int(getenv("WORKERS", "1"))):
        Thread(target=worker, daemon=True).start()
    app.run(port=int(getenv("PORT", "5000")))
