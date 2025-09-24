import dotenv
import flask
import goose3
import io
import logging
import os
import queue
import langdetect
import requests
import spacy
import spacy.cli
import sys
import threading
import wikipediaapi
import wordcloud

dotenv.load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    logging.critical(
        "A variável de ambiente TELEGRAM_BOT_TOKEN não está definida. Encerrando aplicação."
    )
    sys.exit(1)
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}"

app = flask.Flask(__name__)
processing_queue = queue.Queue()
wiki = wikipediaapi.Wikipedia(
    user_agent="FrankRobot (frank.schlemmermeyer@fatec.sp.gov.br)"
)

nlp_model = {"en": "en_core_web_trf", "pt": "pt_core_news_lg"}
nlp = {}

for lang in nlp_model.keys():
    for _ in range(2):
        try:
            nlp[lang] = spacy.load(nlp_model[lang])
            break
        except OSError as e:
            logging.warning(
                f"Modelo spaCy '{nlp_model[lang]}' não encontrado:\n{e}\nBaixando modelo...\n"
            )
            spacy.cli.download(nlp_model[lang])
        except Exception as e:
            logging.critical(f"Erro inesperado ao carregar modelo spaCy:\n{e}\n")
            sys.exit(1)


@app.route("/", methods=["GET"])
def index():
    """
    Endpoint de teste para verificar se o serviço está rodando.
    """
    return "everything is awesome", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Endpoint que recebe atualizações do Telegram via webhook.
    Coloca a mensagem recebida na fila de processamento.
    """
    if (
        (data := flask.request.get_json(silent=True)) is None
        or (message := data.get("message")) is None
        or (message_from := message.get("from")) is None
    ):
        logging.warning(
            f"Dados recebidos no webhook estão incompletos ou malformados: {data}\n"
        )
        return "ignored", 400

    message_text = message.get("text", "")

    logging.info(
        f'{message_from.get("id", "")} {message_from.get("first_name", "")} {message_from.get("last_name", "")} {message_text}'
    )

    processing_queue.put(
        (
            message["chat"].get("id"),
            message_text,
        )
    )

    return "ok", 200


def worker():
    """
    Worker que processa mensagens da fila, gera nuvem de palavras e responde o usuário.
    """
    while True:
        message_chat_id, message_text = processing_queue.get()

        if message_text == "/start":
            send_text(
                message_chat_id,
                "Envie um link, texto ou tópico para gerar uma nuvem de palavras",
            )
            processing_queue.task_done()
            continue

        response_id = (
            send_text(message_chat_id, "Processando...")
            .json()
            .get("result", {})
            .get("message_id")
        )

        try:
            logging.info(
                f"[chat_id={message_chat_id}] Iniciando processamento da mensagem: {message_text}"
            )
            with io.BytesIO() as image_buffer:
                wordcloud.WordCloud(width=1024, height=1024).generate(
                    preprocessing(get_base_text(message_text))
                ).to_image().save(image_buffer, format="PNG")
                image_buffer.seek(0)
                switch_to_image(message_chat_id, response_id, image_buffer)
            logging.info(
                f"[chat_id={message_chat_id}] Nuvem de palavras enviada com sucesso."
            )
        except ValueError as ve:
            logging.warning(f"[chat_id={message_chat_id}] Erro de valor: {ve}")
            edit_text(message_chat_id, response_id, f"Erro: {ve}")
        except Exception as e:
            logging.exception(
                f"[chat_id={message_chat_id}] Erro inesperado ao gerar nuvem de palavras: {e}"
            )
            edit_text(
                message_chat_id,
                response_id,
                "Erro inesperado ao gerar nuvem de palavras. Tente novamente mais tarde.",
            )
        processing_queue.task_done()


def detect_language(text):
    return (lang := langdetect.detect(text)) if lang in nlp else "en"


def get_base_text(message_text):
    if not message_text or not message_text.strip():
        raise ValueError("Mensagem vazia. Envie um texto, link ou tópico válido.")

    if len(message_text) > 4096:
        raise ValueError(
            "O texto é muito grande. Envie um texto com até 4096 caracteres."
        )

    if nlp["en"](message_text.split()[0])[0].like_url:
        return goose3.Goose().extract(message_text).cleaned_text

    if (wiki_page := wiki.page(message_text[:256])).exists():
        return (
            wiki_page.langlinks[lang := detect_language(message_text)].text
            if lang in wiki_page.langlinks
            else wiki_page.text
        )

    if len(message_text.split()) < 7:
        raise ValueError(
            "Envie um texto mais longo, um link ou um tópico da Wikipedia."
        )

    return message_text


def preprocessing(text):
    """
    Pré-processa o texto, lematizando e filtrando tokens relevantes para a nuvem de palavras.
    """
    return " ".join(
        [
            token.lemma_ if token.pos_ == "PROPN" else token.lemma_.lower()
            for token in nlp[detect_language(text)](text)
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


def send_text(chat_id, text):
    """
    Envia uma mensagem de texto para o chat do Telegram.
    """
    return requests.post(
        f"{TELEGRAM_API_URL}/sendMessage",
        data={"chat_id": f"{chat_id}", "text": f"{text}"},
    )


def edit_text(chat_id, message_id, text):
    """
    Edita uma mensagem de texto já enviada no chat do Telegram.
    """
    return requests.post(
        f"{TELEGRAM_API_URL}/editMessageText",
        data={
            "chat_id": f"{chat_id}",
            "message_id": f"{message_id}",
            "text": f"{text}",
        },
    )


def send_image(chat_id, image):
    """
    Envia uma imagem (nuvem de palavras) para o chat do Telegram.
    """
    return requests.post(
        f"{TELEGRAM_API_URL}/sendPhoto",
        files={"photo": image},
        data={"chat_id": f"{chat_id}"},
    )


def switch_to_image(chat_id, response_id, image):
    """
    Substitui a mensagem de texto por uma imagem no chat do Telegram.
    """
    return requests.post(
        f"{TELEGRAM_API_URL}/editMessageMedia",
        files={"photo": image},
        data={
            "chat_id": f"{chat_id}",
            "message_id": f"{response_id}",
            "media": flask.json.dumps({"type": "photo", "media": "attach://photo"}),
        },
    )


def delete_message(chat_id, message_id):
    """
    Deleta uma mensagem do chat do Telegram.
    """
    return requests.post(
        f"{TELEGRAM_API_URL}/deleteMessage",
        data={"chat_id": f"{chat_id}", "message_id": f"{message_id}"},
    )


if __name__ == "__main__":
    for _ in range(int(os.getenv("WORKERS", "1"))):
        threading.Thread(target=worker, daemon=True).start()
    app.run(port=int(os.getenv("PORT", "5000")))
