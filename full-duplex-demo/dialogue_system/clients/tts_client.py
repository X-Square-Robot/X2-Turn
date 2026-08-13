import requests
import io, wave

from modules.utils.MyTn.cn_tn import TextNorm


class IndexTTS_VLLM:
    def __init__(self, speaker: str = "elva", api_url: str = "http://127.0.0.1:6017/tts"):
        self.speaker = speaker
        self.api_url = api_url
        self.normalizer = TextNorm()
        self.MORE_PUNCT = "'\",;:、，；：\n"

        # TextNorm lazily initializes large rule tables on its first call.
        # Pay that cost during service startup rather than on the first turn.
        try:
            warm_text = self.normalizer("你好").strip() or "你好"
            requests.post(
                self.api_url,
                json={"text": warm_text, "character": self.speaker},
                timeout=(2, 8),
            )
        except Exception as e:
            print(f"[IndexTTS_VLLM] warm-up skipped: {e}")

    def synthesize(
        self, text: str, sample_rate: int = 24000, streaming=None, stop_event=None
    ):
        text = self.normalizer(text).replace("*", "").replace("-", " ").strip()
        if text and text[-1] in self.MORE_PUNCT:
            text = text[:-1]
        if not text:
            return

        data = {"text": text, "character": self.speaker}

        # Prefer streaming endpoint (chunked raw PCM s16le 24k) for low latency
        stream_url = self.api_url.replace("/tts", "/tts_stream")
        try:
            with requests.post(
                stream_url, json=data, stream=True, timeout=(5, 60)
            ) as response:
                if response.status_code == 200:
                    got = False
                    for chunk in response.iter_content(chunk_size=9600):
                        if stop_event and stop_event.is_set():
                            response.close()
                            return
                        if chunk:
                            got = True
                            yield chunk
                    if got:
                        return
        except Exception as e:
            print(f"IndexTTS_VLLM streaming failed, fallback to /tts: {e}")

        # Fallback: whole-file wav endpoint
        try:
            if stop_event and stop_event.is_set():
                return
            response = requests.post(self.api_url, json=data, timeout=(5, 60))
            with io.BytesIO(response.content) as wav_buffer:
                with wave.open(wav_buffer, "rb") as wav_file:
                    if stop_event and stop_event.is_set():
                        return
                    yield wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            print(f"IndexTTS_VLLM inference failed: {e}")
            return
# test
if __name__ == "__main__":
    import time

    tts = IndexTTS_VLLM("elva", api_url="http://127.0.0.1:6017/tts")
    text = "Next time you can go earlier or choose a time with fewer people."
    start_time = time.time()
    wav = tts.synthesize(text)
    end_time = time.time()
    print(f"Response time: {end_time - start_time} seconds")
