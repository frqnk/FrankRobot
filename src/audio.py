import asyncio
import io
import tempfile
import requests
import speech_recognition

from pathlib import Path
from typing import Optional
from gtts import gTTS
from pydub import AudioSegment
from slugify import slugify

class AudioManager:
    _HF_TTS_MODELS = {
        "en": "facebook/mms-tts-eng",
        "pt": "facebook/mms-tts-por",
    }

    def __init__(self, huggingface_token: Optional[str] = None) -> None:
        self._recognizer = speech_recognition.Recognizer()
        self._huggingface_token = huggingface_token or ""

    async def voice_to_text(self, file_path: Path) -> Optional[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._voice_to_text_sync, file_path)

    def _voice_to_text_sync(self, file_path: Path) -> Optional[str]:
        wav_path = file_path.with_suffix(".wav")
        try:
            audio = AudioSegment.from_file(file_path, format="ogg")
        except Exception:
            return None
        audio.export(wav_path, format="wav")

        with speech_recognition.AudioFile(str(wav_path)) as source:
            audio_data = self._recognizer.record(source)
        wav_path.unlink(missing_ok=True)

        try:
            return self._recognizer.recognize_google(audio_data)
        except speech_recognition.UnknownValueError:
            return None
        except speech_recognition.RequestError:
            return None

    async def text_to_voice(
        self, text: str, language: str = "en"
    ) -> Optional[io.BytesIO]:
        audio_bytes = await self._generate_huggingface_tts(text, language)
        if audio_bytes is None:
            audio_bytes = await self._generate_gtts(text, language)
        return audio_bytes

    async def _generate_huggingface_tts(
        self, text: str, language: str
    ) -> Optional[io.BytesIO]:
        if not self._huggingface_token:
            return None

        model = self._HF_TTS_MODELS.get(
            language.split("-")[0], self._HF_TTS_MODELS.get("en")
        )
        headers = {"Authorization": f"Bearer {self._huggingface_token}"}
        payload = {"inputs": text}

        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json=payload,
                timeout=60,
            ),
        )

        if response.status_code != 200 or not response.content:
            return None

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return None

        audio_stream = io.BytesIO(response.content)
        audio_stream.name = f"reply_{slugify(text[:40]) or 'speech'}.mp3"
        audio_stream.seek(0)
        return audio_stream

    async def _generate_gtts(self, text: str, language: str) -> Optional[io.BytesIO]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._gtts_sync, text, language)

    def _gtts_sync(self, text: str, language: str) -> Optional[io.BytesIO]:
        try:
            tts = gTTS(text=text, lang=language.split("-")[0])
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                tts.save(temp_file.name)
                temp_path = Path(temp_file.name)

            with temp_path.open("rb") as audio_file:
                data = audio_file.read()
            temp_path.unlink(missing_ok=True)

            stream = io.BytesIO(data)
            stream.name = f"reply_{slugify(text[:40]) or 'speech'}.mp3"
            stream.seek(0)
            return stream
        except Exception:
            return None
