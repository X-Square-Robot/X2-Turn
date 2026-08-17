import re
import os
import time
import soxr
import queue
import base64
import logging
import threading
import uuid
import json
import asyncio
import numpy as np
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from clients.tts_client import IndexTTS_VLLM
from clients.llm_client import QwenLLM_stream
from clients.vad_client import TurnTaking
from modules.utils.backchannel_utils import check_backchannel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()


# Headers required for SharedArrayBuffer
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


# Global event loop reference for thread-safe websocket sending
main_loop = None


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()


class Config:
    """Global configuration constants."""

    SAMPLE_RATE = 16000
    # Voxtral HF bridge shares one GPU model; keep pool small.
    VAD_POOL_SIZE = 4
    PORT = int(os.environ.get("DEMO_PORT", "8443"))


class ChatSession:
    """Manages the full lifecycle of a single user session."""

    def __init__(self, client_id, vad_instance, websocket: WebSocket):
        self.client_id = client_id
        self.vad = vad_instance
        self.websocket = websocket
        self.lock = threading.Lock()
        self.send_lock = asyncio.Lock()
        self._stop_event = threading.Event()  # Internal event to signal interruption
        self.is_active = True
        self.generation_epoch = 0
        self.generation_in_progress = False
        self.metrics = {}

        # Audio config
        self.input_sample_rate = Config.SAMPLE_RATE

        # State for pending message commitment (handling interruption)
        self.pending_message = None
        self.pending_audio_duration = 0.0
        self.pending_start_time = None
        self.interruption_time = None
        self.bot_speaking = False

    @property
    def stop_event(self):
        return self._stop_event

    def begin_generation(self, accepted_at):
        """Atomically invalidate the old generation and create a new epoch."""
        with self.lock:
            self._stop_event.set()
            self.generation_epoch += 1
            epoch = self.generation_epoch
            self._stop_event = threading.Event()
            self.bot_speaking = False
            self.generation_in_progress = True
            pending = (
                self.pending_message,
                self.pending_audio_duration,
                self.pending_start_time,
                self.interruption_time,
            )
            self.pending_message = None
            self.pending_audio_duration = 0.0
            self.pending_start_time = None
            self.interruption_time = None
            self.metrics[epoch] = {
                "accepted_at": accepted_at,
                "turn_id": f"{self.client_id[:8]}-{epoch}",
            }
            for old_epoch in sorted(self.metrics)[:-8]:
                self.metrics.pop(old_epoch, None)
            return epoch, self._stop_event, pending

    def is_current(self, epoch, stop_event=None):
        with self.lock:
            return (
                self.is_active
                and self.generation_epoch == epoch
                and not (stop_event and stop_event.is_set())
            )

    def interrupt(self, reason="barge_in"):
        """Interrupts current inference or audio playback."""
        with self.lock:
            self.bot_speaking = False
            self.generation_in_progress = False
            self._stop_event.set()
            self.generation_epoch += 1
            epoch = self.generation_epoch
        try:
            self.vad.set_bot_speaking(False)
        except Exception:
            pass
        emit_to_room(
            self.client_id,
            "stop_audio",
            {"message": "interrupt", "reason": reason, "epoch": epoch},
        )
        emit_to_room(self.client_id, "circle_status", {"status": "LISTENING"})
        return epoch

    def pause(self):
        """Pauses audio playback."""
        emit_to_room(self.client_id, "pause_audio", {"message": "pause"})
        emit_to_room(self.client_id, "circle_status", {"status": "LISTENING"})

class SessionManager:
    """Thread-safe manager for active chat sessions."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def create_session(self, client_id, vad_instance, websocket: WebSocket):
        with self._lock:
            session = ChatSession(client_id, vad_instance, websocket)
            self.sessions[client_id] = session
            return session

    def get_session(self, client_id) -> Optional[ChatSession]:
        return self.sessions.get(client_id)

    def remove_session(self, client_id):
        with self._lock:
            return self.sessions.pop(client_id, None)


class VADModelPool:
    """Object pool for VAD instances to optimize memory and startup time."""

    def __init__(self, model_cls, size=Config.VAD_POOL_SIZE):
        self.pool = queue.Queue(maxsize=size)
        logger.info(f"Initializing VAD Pool with {size} instances...")
        for _ in range(size):
            # Initialize instances without specific callbacks (bound during acquisition)
            instance = model_cls(
                status_callback=None, transcription_callback=None, circle_callback=None
            )
            self.pool.put(instance)

    def acquire(self):
        """Retrieve a VAD instance from the pool."""
        return self.pool.get(block=True)

    def release(self, instance):
        """Reset and return the instance back to the pool."""
        try:
            if hasattr(instance, "reset"):
                instance.reset()
        except Exception as e:
            logger.warning(f"VAD pool release reset failed: {e}")
        # Clear callbacks to prevent memory leaks or stale context
        instance.status_callback = None
        instance.transcription_callback = None
        instance.circle_callback = None
        self.pool.put(instance)


async def emit_ws_event(ws: WebSocket, event: str, data):
    """Send JSON event directly on a WebSocket (before session exists)."""
    if ws.client_state != WebSocketState.CONNECTED:
        return
    await ws.send_text(json.dumps({"event": event, "data": data}))


# ==== Global Singleton Initialization ====
vad_pool = VADModelPool(TurnTaking)
session_manager = SessionManager()
llm = QwenLLM_stream()
tts = IndexTTS_VLLM(api_url=os.environ.get("TTS_API_URL", "http://127.0.0.1:6017/tts"))
asr = None
print("System initialized: VAD Pool, LLM client, TTS client ready.")


def emit_to_room(client_id, event, data):
    """Helper function to safely emit WebSocket messages to a specific client."""
    session = session_manager.get_session(client_id)
    if not session or not main_loop:
        return

    ws = session.websocket

    if ws.client_state != WebSocketState.CONNECTED:
        return

    # Helper wrapper to run async send in the main loop
    async def _send():
        try:
            async with session.send_lock:
                if ws.client_state != WebSocketState.CONNECTED:
                    return
                if event == "audio_chunk":
                    await ws.send_bytes(data)
                else:
                    message = json.dumps({"event": event, "data": data})
                    await ws.send_text(message)
        except Exception as e:
            if ws.client_state == WebSocketState.CONNECTED:
                logger.debug(f"WebSocket send ended for {client_id}: {e}")

    asyncio.run_coroutine_threadsafe(_send(), main_loop)


def emit_audio_chunk(client_id, epoch, pcm):
    """Serialize raw PCM sends and drop stale generations before writing."""
    session = session_manager.get_session(client_id)
    if not session or not main_loop or not session.is_current(epoch):
        return
    ws = session.websocket

    async def _send():
        current = session_manager.get_session(client_id)
        if not current:
            return
        try:
            async with current.send_lock:
                if (
                    current.is_current(epoch)
                    and ws.client_state == WebSocketState.CONNECTED
                ):
                    await ws.send_bytes(pcm)
        except Exception as e:
            if ws.client_state == WebSocketState.CONNECTED:
                logger.debug(f"Audio send ended for {client_id}: {e}")

    asyncio.run_coroutine_threadsafe(_send(), main_loop)


def commit_pending_message(client_id, pending, interrupted_at=None):
    """Commit the portion of the previous response that was likely heard."""
    message, duration, started_at, saved_interruption = pending
    if not message:
        return
    cutoff_at = interrupted_at or saved_interruption
    final_message = message
    if cutoff_at and started_at and duration > 0:
        ratio = min(max((cutoff_at - started_at) / duration, 0.0), 1.0)
        final_message = message[: int(len(message) * ratio)]
    if final_message:
        llm.add_message(client_id, "assistant", final_message)


def drain_pending_message(session, interrupted_at=None):
    with session.lock:
        pending = (
            session.pending_message,
            session.pending_audio_duration,
            session.pending_start_time,
            session.interruption_time,
        )
        session.pending_message = None
        session.pending_audio_duration = 0.0
        session.pending_start_time = None
        session.interruption_time = interrupted_at
    commit_pending_message(session.client_id, pending, interrupted_at)


def pipeline_worker(client_id, audio_segment, sample_rate, accepted_at=None):
    """
    Main processing pipeline: ASR -> LLM -> TTS.
    Runs in a background thread for each detected utterance.
    """
    session = session_manager.get_session(client_id)
    if not session:
        return

    try:
        accepted_at = accepted_at or time.perf_counter()
        epoch, current_stop_event, previous_pending = session.begin_generation(
            accepted_at
        )
        turn_id = session.metrics[epoch]["turn_id"]
        emit_to_room(
            client_id,
            "stop_audio",
            {"message": "new generation", "reason": "new_turn", "epoch": epoch},
        )
        commit_pending_message(client_id, previous_pending, time.time())

        # 1. ASR Phase (Automatic Speech Recognition)
        if asr is None:
            # Fallback if ASR is handled by internal TurnTaking module
            asr_text = (
                audio_segment if isinstance(audio_segment, str) else "Voice Detected"
            )
        else:
            asr_text = asr.recognize(audio_segment, sample_rate)

        if not asr_text.strip():
            return

        # # 2. Backchannel Detection (vad already handles this, but double-check here)
        # # Checks for filler words or short backchannels that shouldn't trigger a full response
        # if check_backchannel(asr_text):
        #     emit_to_room(client_id, "resume_audio", {"message": "backchannel detected"})
        #     return

        if not session.is_current(epoch, current_stop_event):
            return

        asr_at = time.perf_counter()
        session.metrics[epoch]["asr_at"] = asr_at
        logger.info(
            f"[LATENCY] turn={turn_id} stage=asr "
            f"accepted_to_asr_ms={(asr_at - accepted_at) * 1000:.1f}"
        )
        logger.info(f"[{client_id}] ASR result: {asr_text}")
        emit_to_room(
            client_id,
            "user_transcription",
            {"text": asr_text, "epoch": epoch, "turn_id": turn_id},
        )
        emit_to_room(client_id, "circle_status", {"status": "THINKING"})

        llm.add_message(client_id, "user", asr_text)
        llm_reply_gen = llm.generate_with_history(
            client_id, stop_event=current_stop_event
        )

        # LLM producer runs independently so TTS cannot stall the HTTP stream.
        text_queue = queue.Queue(maxsize=8)
        producer_done = object()
        message_parts = []
        parts_lock = threading.Lock()

        def put_text(item):
            while session.is_current(epoch, current_stop_event):
                try:
                    text_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def llm_producer():
            first = True
            try:
                source = (
                    llm_reply_gen
                    if hasattr(llm_reply_gen, "__iter__")
                    and not isinstance(llm_reply_gen, str)
                    else [llm_reply_gen]
                )
                for chunk in source:
                    if not session.is_current(epoch, current_stop_event):
                        break
                    now = time.perf_counter()
                    if first:
                        first = False
                        session.metrics[epoch]["llm_first_at"] = now
                        logger.info(
                            f"[LATENCY] turn={turn_id} stage=llm_first "
                            f"asr_to_llm_ms={(now - asr_at) * 1000:.1f}"
                        )
                    logger.info(f"[{client_id}] LLM Chunk: {chunk}")
                    with parts_lock:
                        message_parts.append(chunk)
                    emit_to_room(
                        client_id,
                        "text_response",
                        {"text": chunk, "epoch": epoch, "turn_id": turn_id},
                    )
                    if not put_text(chunk):
                        break
            finally:
                put_text(producer_done)

        producer_thread = threading.Thread(target=llm_producer, daemon=True)
        producer_thread.start()

        spoken_parts = []
        first_emit_time = None
        total_audio_duration = 0.0
        first_tts_request_at = None

        while session.is_current(epoch, current_stop_event):
            try:
                chunk = text_queue.get(timeout=0.1)
            except queue.Empty:
                if not producer_thread.is_alive():
                    break
                continue
            if chunk is producer_done:
                break
            spoken_parts.append(chunk)
            tts_request_at = time.perf_counter()
            if first_tts_request_at is None:
                first_tts_request_at = tts_request_at
                session.metrics[epoch]["tts_request_at"] = tts_request_at

            for wav_chunk in tts.synthesize(
                chunk, streaming=True, stop_event=current_stop_event
            ):
                if not session.is_current(epoch, current_stop_event):
                    break

                if first_emit_time is None:
                    first_emit_time = time.time()
                    first_pcm_at = time.perf_counter()
                    session.metrics[epoch]["first_pcm_at"] = first_pcm_at
                    logger.info(
                        f"[LATENCY] turn={turn_id} stage=tts_first_pcm "
                        f"tts_first_pcm_ms={(first_pcm_at - first_tts_request_at) * 1000:.1f} "
                        f"accepted_to_pcm_ms={(first_pcm_at - accepted_at) * 1000:.1f}"
                    )
                    with session.lock:
                        session.bot_speaking = True
                        session.vad.set_bot_speaking(True)

                # Calculate audio duration: bytes / (sample_rate * channels * bytes_per_sample)
                # Assuming 24k sample rate, 1 channel, 16-bit (2 bytes) = 48000 bytes/sec
                total_audio_duration += len(wav_chunk) / 48000.0
                emit_audio_chunk(client_id, epoch, wav_chunk)

                # Keep a continuously available snapshot for barge-in commit.
                with session.lock:
                    if session.generation_epoch == epoch:
                        session.pending_message = "".join(spoken_parts)
                        session.pending_audio_duration = total_audio_duration
                        session.pending_start_time = first_emit_time

        producer_thread.join(timeout=1.0)
        if session.is_current(epoch, current_stop_event) and total_audio_duration > 0:
            with parts_lock:
                complete_message = "".join(message_parts)
            with session.lock:
                session.pending_message = complete_message
                session.pending_audio_duration = total_audio_duration
                session.pending_start_time = first_emit_time
                session.interruption_time = None
                session.generation_in_progress = False
            emit_to_room(
                client_id,
                "audio_generation_done",
                {"epoch": epoch, "turn_id": turn_id},
            )
        elif session.is_current(epoch, current_stop_event):
            with session.lock:
                session.generation_in_progress = False

    except Exception as e:
        logger.error(f"Error in pipeline for {client_id}: {e}", exc_info=True)
        with session.lock:
            if "epoch" in locals() and session.generation_epoch == epoch:
                session.generation_in_progress = False


# ==== WebSocket Endpoint & Processing ====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    logger.info(f"New connection: {client_id}")

    session = None

    try:
        # Initial Handshake / Setup
        await emit_ws_event(
            websocket,
            "vad_loading",
            {"state": "loading", "message": "Acquiring model resources..."},
        )

        loop = asyncio.get_running_loop()
        vad_instance = await loop.run_in_executor(None, vad_pool.acquire)

        # Bind Callbacks
        vad_instance.status_callback = lambda s, m: emit_to_room(
            client_id, "vad_status", {"state": s, "message": m}
        )
        vad_instance.transcription_callback = lambda t: emit_to_room(
            client_id, "user_transcription", {"text": t}
        )
        vad_instance.circle_callback = lambda s: emit_to_room(
            client_id, "circle_status", {"status": s}
        )

        session = session_manager.create_session(client_id, vad_instance, websocket)

        # Now emission works via session_manager lookups
        emit_to_room(client_id, "connect_ack", {"client_id": client_id})
        emit_to_room(
            client_id,
            "vad_loading",
            {"state": "ready", "message": "Model loaded, ready to experience"},
        )
        logger.info(f"Session initialized for {client_id}")

        while True:
            # Receive Message
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {client_id}")
                break
            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client disconnected: {client_id}")
                break

            if "bytes" in message and message["bytes"]:
                # Binary Audio Data
                data = message["bytes"]
                # Convert buffer to float32
                audio_chunk = (
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                current_sr = session.input_sample_rate
                if current_sr != Config.SAMPLE_RATE:
                    audio_chunk = soxr.resample(
                        audio_chunk, current_sr, Config.SAMPLE_RATE, quality="VHQ"
                    )

                # The bridge client uses a synchronous WebSocket round trip.
                # Keep it off the FastAPI event loop so outbound PCM and
                # control messages are not delayed by incoming mic frames.
                def process_vad_chunk():
                    with session.lock:
                        return session.vad.process(
                            audio_chunk, bot_speaking=session.bot_speaking
                        )

                segment = await loop.run_in_executor(None, process_vad_chunk)

                # Forward fine-grained turn state to frontend timeline viz
                detail = getattr(session.vad, "last_detail", None)
                if detail:
                    emit_to_room(
                        client_id,
                        "turn_viz",
                        {
                            "turn_class": detail.get("turn_class", "idle"),
                            "state": detail.get("state", ""),
                            "event": detail.get("event"),
                            "reason": detail.get("reason"),
                            "note": detail.get("note"),
                            "acoustic_active": detail.get(
                                "acoustic_active", False
                            ),
                            "acoustic_rms": detail.get("acoustic_rms"),
                        },
                    )

                if segment is not None:
                    if isinstance(segment, list) and segment[0] is None:
                        # Interrupt only audio that is already playing. A
                        # ``turn_end`` or early ``speaking`` event while the
                        # LLM/TTS is still generating is commonly residual
                        # loudspeaker echo; a completed user turn will start a
                        # new generation through the normal path below.
                        turn_class = (detail or {}).get("turn_class")
                        with session.lock:
                            should_interrupt = (
                                session.bot_speaking and turn_class == "speaking"
                            )
                        if should_interrupt:
                            interrupted_at = time.time()
                            logger.info(
                                f"[{client_id}] Barge-in: turn_class={turn_class} "
                                f"reason={(detail or {}).get('reason')}"
                            )
                            session.interrupt(reason="barge_in")
                            drain_pending_message(session, interrupted_at)
                    else:
                        # Complete Utterance
                        accepted_at = time.perf_counter()
                        threading.Thread(
                            target=pipeline_worker,
                            args=(
                                client_id,
                                segment,
                                Config.SAMPLE_RATE,
                                accepted_at,
                            ),
                            daemon=True,
                        ).start()

            elif "text" in message and message["text"]:
                # JSON Control Message
                try:
                    payload = json.loads(message["text"])
                    event = payload.get("event")

                    if event == "duplex_stop":
                        interrupted_at = time.time()
                        session.interrupt(reason="manual_stop")
                        drain_pending_message(session, interrupted_at)
                        logger.info(f"Session manually stopped by client: {client_id}")

                    elif event == "config_audio":
                        # Client sending sample rate configuration
                        sr_data = payload.get("data", {})
                        sr = sr_data.get("sample_rate")
                        if sr:
                            session.input_sample_rate = int(sr)
                            logger.info(f"[{client_id}] Sample rate set to {sr}")

                    elif event == "playback_started":
                        event_data = payload.get("data", {})
                        epoch = int(event_data.get("epoch", -1))
                        metrics = session.metrics.get(epoch)
                        if metrics and "playback_started_at" not in metrics:
                            now = time.perf_counter()
                            metrics["playback_started_at"] = now
                            logger.info(
                                f"[LATENCY] turn={metrics['turn_id']} "
                                f"stage=browser_playback "
                                f"accepted_to_playback_ms="
                                f"{(now - metrics['accepted_at']) * 1000:.1f}"
                            )

                    elif event == "playback_ended":
                        event_data = payload.get("data", {})
                        epoch = int(event_data.get("epoch", -1))
                        if session.is_current(epoch):
                            with session.lock:
                                session.bot_speaking = False
                            session.vad.set_bot_speaking(False)
                            drain_pending_message(session)

                except json.JSONDecodeError:
                    logger.warning(f"[{client_id}] Received invalid JSON")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if session:
            session.is_active = False
            session.stop_event.set()
            vad_pool.release(session.vad)
            session_manager.remove_session(client_id)
        logger.info(f"Session cleaned up: {client_id}")


# ==== Static Resource Routing ====
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # Public/non-localhost access needs HTTPS for getUserMedia + SharedArrayBuffer.
    ssl_cert = os.environ.get("DEMO_SSL_CERT")
    ssl_key = os.environ.get("DEMO_SSL_KEY")
    if not ssl_cert or not ssl_key:
        default_dir = os.path.join(os.path.dirname(__file__), "..", "certs")
        cand_cert = os.path.join(default_dir, "cert.pem")
        cand_key = os.path.join(default_dir, "key.pem")
        if os.path.isfile(cand_cert) and os.path.isfile(cand_key):
            ssl_cert, ssl_key = cand_cert, cand_key

    bind_host = os.environ.get("DEMO_BIND_HOST", "127.0.0.1")
    scheme = "https" if ssl_cert and ssl_key else "http"
    logger.info(f"Server starting on {scheme}://{bind_host}:{Config.PORT}")
    uvicorn.run(
        app,
        host=bind_host,
        port=Config.PORT,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
    )
