import re
import os
import requests
from itertools import groupby
from modules.utils.text_utils import split_cn_en


class QwenLLM_stream:
    PUNCT = r"[.!?;:。！？；：\n]"
    MORE_PUNCT = r"[,.!?;:、，。！？；：\n]"
    MIN_LEN_FOR_SEG = 5
    FIRST_SEG_MAX_LEN = 8
    MAX_BUFFER_LEN = 24
    MAX_BUFFER_SESSION = 5

    SYSTEM_PROMPT = (
        "You are a gentle and natural voice conversation assistant."
        "Your name is Elva."
        "You are communicating with the user via speech; please respond in a natural, brief, and colloquial manner."
        "Do not output extra explanations, do not use lists or markdown format."
        "Keep the conversation coherent, just like talking in reality."
        "If the user shows agreement, affirmation or backchannels, continue speaking naturally based on the previous context."
        "If the user asks you to stop, output nothing."
    )

    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.environ.get(
            "LLM_API_URL", "http://localhost:6007/chat"
        )
        self.sessions = {}

    def get_session(self, client_id: int):
        if client_id not in self.sessions:
            self.sessions[client_id] = []
        return self.sessions[client_id]

    def add_message(self, client_id: int, role: str, content: str):
        session = self.get_session(client_id)
        session.append({"role": role, "content": content})

        merged = []
        for role_key, group in groupby(session, key=lambda x: x["role"]):
            contents = [msg["content"] for msg in group]
            merged.append({"role": role_key, "content": "\n".join(contents)})

        self.sessions[client_id] = merged[-self.MAX_BUFFER_SESSION :]

    def pop_segment(self, buffer: str, first_segment: bool = False):
        if len(split_cn_en(buffer)) < self.MIN_LEN_FOR_SEG:
            return None, buffer

        # Emit the earliest natural clause that is long enough. The first
        # clause accepts commas so TTS can start before a full sentence ends.
        punct = self.MORE_PUNCT if first_segment else self.PUNCT
        for match in re.finditer(punct, buffer):
            idx = match.end()
            if len(split_cn_en(buffer[:idx])) >= self.MIN_LEN_FOR_SEG:
                return buffer[:idx], buffer[idx:]

        if first_segment and len(split_cn_en(buffer)) >= self.FIRST_SEG_MAX_LEN:
            return buffer, ""

        # buffer too long, force cut
        if len(split_cn_en(buffer)) >= self.MAX_BUFFER_LEN:
            for match in re.finditer(self.MORE_PUNCT, buffer):
                idx = match.end()
                if len(split_cn_en(buffer[:idx])) >= self.MIN_LEN_FOR_SEG:
                    return buffer[:idx], buffer[idx:]
            return buffer, ""

        return None, buffer

    def generate_with_history(self, client_id: int, stop_event=None):
        messages = self.get_session(client_id)
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages

        try:
            response = requests.post(
                self.api_url, json={"messages": conversation}, stream=True, timeout=60
            )

            buffer = ""
            emitted_any = False
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if stop_event and stop_event.is_set():
                    response.close()
                    break
                if not chunk:
                    continue

                buffer += chunk

                while True:
                    seg, buffer = self.pop_segment(buffer, first_segment=not emitted_any)
                    if seg:
                        emitted_any = True
                        yield seg
                    else:
                        break

            if buffer.strip() and not (stop_event and stop_event.is_set()):
                yield buffer.strip()

        except Exception as e:
            print(f"Qwen API call failed: {e}")
            yield "Sorry, I cannot answer right now."


# test
if __name__ == "__main__":
    import time

    llm = QwenLLM_stream(api_url="http://localhost:6007/chat")
    text = "Hello, can you introduce yourself in detail?"
    llm.add_message(0, "user", text)
    start_time = time.time()
    for reply in llm.generate_with_history(0):
        end_time = time.time()
        print(f"Response time: {end_time - start_time} seconds | {reply}")
        start_time = time.time()
