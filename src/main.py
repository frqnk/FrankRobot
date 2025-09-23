import flask, requests, dotenv, os, threading, queue, wordcloud, goose3, io, spacy, spacy.cli, wikipediaapi

dotenv.load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}"

app = flask.Flask(__name__)
processing_queue = queue.Queue()
wiki = wikipediaapi.Wikipedia(user_agent="FrankRobot/0.1")

nlp_model_sm = "en_core_web_sm"
nlp_model_lg = "en_core_web_md"

try:
    nlp_sm = spacy.load(nlp_model_sm)
    nlp_lg = spacy.load(nlp_model_lg)
except:
    spacy.cli.download(nlp_model_sm)
    spacy.cli.download(nlp_model_lg)
    nlp_sm = spacy.load(nlp_model_sm)
    nlp_lg = spacy.load(nlp_model_lg)


@app.route("/webhook", methods=["POST"])
def webhook():
    data = flask.request.get_json(silent=True)
    message = data.get("message", {})
    message_from = message.get("from", {})
    message_text = message.get("text", "")

    print(
        f'{message_from.get("id", "")} {message_from.get("first_name", "")} {message_from.get("last_name", "")} {message_text}'
    )

    processing_queue.put(
        (
            message.get("chat", {}).get("id"),
            message_text,
        )
    )

    return "ok"


def worker():
    while True:
        chat_id, message_text = processing_queue.get()

        if message_text == "/start":
            send_text(
                chat_id,
                "Envie um link, texto ou tópico para gerar uma nuvem de palavras",
            )
            processing_queue.task_done()
            continue

        response_id = (
            send_text(chat_id, "Processando...")
            .json()
            .get("result", {})
            .get("message_id")
        )

        try:
            if len(message_text.split()) == 1:
                raise Exception()
            cleaned_text = preprocessing(
                goose3.Goose().extract(message_text).cleaned_text
                if nlp_sm(message_text)[0].like_url
                else (
                    message_text
                    if not wiki.page(message_text[:256]).exists()
                    else wiki.page(message_text[:256]).text
                )
            )
            with io.BytesIO() as image_buffer:
                wordcloud.WordCloud(width=1024, height=1024).generate(
                    cleaned_text
                ).to_image().save(image_buffer, format="PNG")
                image_buffer.seek(0)
                switch_to_image(chat_id, response_id, image_buffer)
        except Exception as e:
            print(e)
            edit_text(chat_id, response_id, "Erro ao gerar nuvem de palavras")
        processing_queue.task_done()


def preprocessing(text):
    doc = nlp_lg(text)
    tokens = [
        token.lemma_ if token.pos_ == "PROPN" else token.lemma_.lower()
        for token in doc
        if (
            token.is_alpha
            and token.pos_ not in {"PRON", "DET", "PART", "AUX"}
            and not token.is_stop
            and not token.is_punct
            and not token.like_url
            and not token.like_email
        )
    ]
    return " ".join(tokens)


def send_text(chat_id, text):
    return requests.post(
        f"{TELEGRAM_API_URL}/sendMessage", data={"chat_id": chat_id, "text": text}
    )


def edit_text(chat_id, message_id, text):
    return requests.post(
        f"{TELEGRAM_API_URL}/editMessageText",
        data={"chat_id": chat_id, "message_id": message_id, "text": text},
    )


def send_image(chat_id, image):
    return requests.post(
        f"{TELEGRAM_API_URL}/sendPhoto",
        files={"photo": image},
        data={"chat_id": chat_id},
    )


def switch_to_image(chat_id, response_id, image):
    return requests.post(
        f"{TELEGRAM_API_URL}/editMessageMedia",
        files={"photo": image},
        data={
            "chat_id": chat_id,
            "message_id": response_id,
            "media": flask.json.dumps({"type": "photo", "media": "attach://photo"}),
        },
    )


def delete_message(chat_id, message_id):
    return requests.post(
        f"{TELEGRAM_API_URL}/deleteMessage",
        data={"chat_id": chat_id, "message_id": message_id},
    )


if __name__ == "__main__":
    for _ in range(int(os.getenv("WORKERS", "4"))):
        threading.Thread(target=worker, daemon=True).start()
    app.run(port=int(os.getenv("PORT", "5000")))
