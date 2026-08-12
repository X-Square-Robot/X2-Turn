import os, sys
import io
import re
import time
import uuid
import json
import threading
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
import argparse
from omegaconf import OmegaConf

from clients.tts_client import IndexTTS_VLLM
from clients.llm_client import QwenLLM_stream
from clients.vad_client import TurnTaking
from modules.utils.backchannel_utils import check_backchannel
from modules.utils.text_utils import split_cn_en


def split_into_segments(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"([。！？.!?])", text)
    segments, current = [], ""
    for part in parts:
        if not part.strip():
            continue
        current += part
        if re.match(r"[。！？.!?]", part):
            segments.append(current.strip())
            current = ""
    if current.strip():
        segments.append(current.strip())
    return segments


def decode_pcm_bytes(pcm_bytes: bytes, target_sr: int, cache: dict) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)

    # Int16 PCM to float32 normalized
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if audio.size == 0:
        return audio

    waveform = torch.from_numpy(audio).unsqueeze(0)

    # Assume source is 24000 as per tts_client default
    src_sr = 24000
    if src_sr != target_sr:
        if src_sr not in cache:
            cache[src_sr] = torchaudio.transforms.Resample(
                orig_freq=src_sr, new_freq=target_sr
            )
        waveform = cache[src_sr](waveform)
    return waveform.squeeze(0).to(dtype=torch.float32).numpy()


def decode_wav_bytes(wav_bytes: bytes, target_sr: int, cache: dict) -> np.ndarray:
    if not wav_bytes:
        return np.zeros(0, dtype=np.float32)
    try:
        with io.BytesIO(wav_bytes) as buf:
            waveform, sr = torchaudio.load(buf)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        if sr not in cache:
            cache[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = cache[sr](waveform)
    return waveform.squeeze(0).to(dtype=torch.float32).numpy()


class OfflineDuplexSimulator:
    def __init__(
        self,
        vad_api,
        llm_api,
        tts_api,
        config,
    ):
        self.vad = vad_api
        self.llm = llm_api
        self.tts = tts_api
        self.sample_rate = config.infer_config.input.sample_rate
        self.chunk_size = config.infer_config.input.chunk_size
        self.chunk_duration = self.chunk_size / self.sample_rate
        self.lock = threading.Lock()
        self._resamplers = {}
        self.developer_mode = bool(
            getattr(config.infer_config, "developer_mode", False)
        )
        self.states = []  # To store duplug states
        self.time_info = []  # To store latency metrics

        # State for pending message commitment
        self.pending_message = None
        self.pending_audio_duration = 0.0
        self.pending_start_sample = None
        self.interruption_sample = None

        self.reset_conversation()

    def reset_conversation(self):
        if hasattr(self, "client_id") and hasattr(self.llm, "sessions"):
            self.llm.sessions.pop(self.client_id, None)
        self.client_id = uuid.uuid4().hex
        if hasattr(self.llm, "sessions"):
            self.llm.sessions.pop(self.client_id, None)
        self.assistant_segments: List[dict] = []
        self.assistant_end_sample = 0
        self.stop_event: Optional[threading.Event] = None
        self.generation_threads: List[threading.Thread] = []
        self.user_audio: Optional[np.ndarray] = None
        self.user_audio_with_padding: Optional[np.ndarray] = None
        self._backlog_sec = 0.0

        self.pending_message = None
        self.pending_audio_duration = 0.0
        self.pending_start_sample = None
        self.interruption_sample = None
        self.bot_speaking = False

        if hasattr(self.vad, "reset"):
            self.vad.reset()
        if hasattr(self.vad, "set_bot_speaking"):
            self.vad.set_bot_speaking(False)

    def _debug(self, msg):
        if self.developer_mode:
            print(msg)

    def run(
        self,
        wav_path: str,
        out_wav_path: Optional[str],
        dialogue_wav_path: Optional[str],
        state_path: Optional[str] = None,
        time_info_path: Optional[str] = None,
    ):
        self.reset_conversation()
        self.states = []  # Reset for this file
        self.time_info = []  # Reset for this file
        result = self._process_file(wav_path)
        self._wait_for_generators()

        assistant_wave = self._render_assistant_wave()
        assistant_wave = np.clip(assistant_wave, -1.0, 1.0)

        dialogue_wave = self._render_dialogue_wave(assistant_wave)
        dialogue_wave = np.clip(dialogue_wave, -1.0, 1.0)

        out_path = out_wav_path or wav_path.replace(".wav", "_assistant.wav")
        dialogue_path = dialogue_wav_path or wav_path.replace(".wav", "_dialogue.wav")

        self._write_wav(out_path, assistant_wave)
        self._write_wav(dialogue_path, dialogue_wave)

        if state_path:
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(self.states, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[OfflineSimulator] Failed to save states: {e}")

        if time_info_path:
            try:
                with open(time_info_path, "w", encoding="utf-8") as f:
                    json.dump(self.time_info, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[OfflineSimulator] Failed to save time info: {e}")

        return result

    def _is_bot_speaking_at(self, sample: int) -> bool:
        """True when assistant TTS covers ``sample`` on the shared timeline."""
        with self.lock:
            for seg in self.assistant_segments:
                start = seg["start"]
                end = start + len(seg["audio"])
                if start <= sample < end:
                    return True
        return False

    def _sync_bot_speaking(self, sample: int) -> bool:
        speaking = self._is_bot_speaking_at(sample)
        if speaking != self.bot_speaking:
            self.bot_speaking = speaking
            if hasattr(self.vad, "set_bot_speaking"):
                self.vad.set_bot_speaking(speaking)
        return speaking

    def _load_wav_mono(self, wav_path: str) -> np.ndarray:
        try:
            data, sr = sf.read(wav_path, dtype="float32")
        except Exception:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            data = waveform.squeeze(0).numpy()
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            data = resampler(torch.from_numpy(data).unsqueeze(0)).squeeze(0).numpy()
        return np.ascontiguousarray(data, dtype=np.float32)

    def _process_file(self, wav_path: str):
        audio = self._load_wav_mono(wav_path)
        self.user_audio = audio.copy()
        self.user_audio_with_padding = audio.copy()

        total = len(audio)
        for offset in range(0, total, self.chunk_size):
            chunk = audio[offset : offset + self.chunk_size]
            effective_samples = len(chunk)
            if effective_samples < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - effective_samples))

            if self._backlog_sec < 0.0:
                sleep_time = -self._backlog_sec
                time.sleep(sleep_time)
                self._backlog_sec = 0.0

            t0 = time.time()
            bot = self._sync_bot_speaking(offset)
            result = self.vad.process(chunk.astype(np.float32), bot_speaking=bot)
            elapsed = time.time() - t0 + max(0.0, self._backlog_sec)
            self._backlog_sec = elapsed - self.chunk_duration
            # print(
            #     f"[OfflineSimulator] VAD processing time: {elapsed:.3f}s, backlog: {self._backlog_sec:.3f}s"
            # )

            current_sample = int(
                min(offset + effective_samples, total) + elapsed * self.sample_rate
            )
            self._record_vad_result(result, offset, current_sample)
            self._handle_vad_result(result, current_sample)

        if self._backlog_sec > 10:
            return False
        else:
            return True

        # silence = np.zeros(self.chunk_size, dtype=np.float32)
        # for _ in range(10):
        #     current_sample = total + (_ + 1) * self.chunk_size
        #     result = self.vad.process(silence)
        #     self._handle_vad_result(result, current_sample)

    def _record_vad_result(self, result, offset: int, current_sample: int):
        current_state = None

        if isinstance(result, list) and result and result[0] is None:
            current_state = {
                "state": "nonidle",
                "timestamp": [
                    offset / self.sample_rate,
                    (offset + self.chunk_size) / self.sample_rate,
                ],
                "current_time": current_sample / self.sample_rate,
            }
        elif isinstance(result, str):
            text = result.strip()
            if text and not check_backchannel(text):
                current_state = {
                    "state": "speak",
                    "transcript": text,
                    "timestamp": [
                        offset / self.sample_rate,
                        (offset + self.chunk_size) / self.sample_rate,
                    ],
                    "current_time": current_sample / self.sample_rate,
                }

        if not current_state:
            current_state = {
                "state": "idle",
                "timestamp": [
                    offset / self.sample_rate,
                    (offset + self.chunk_size) / self.sample_rate,
                ],
                "current_time": current_sample / self.sample_rate,
            }
        self.states.append(current_state)

    def _handle_vad_result(self, result, current_sample: int):
        if result is None:
            return
        if isinstance(result, list) and result and result[0] is None:
            # Set interruption time for the currently active response
            if self.interruption_sample is None:
                self.interruption_sample = current_sample
            self._request_interruption(current_sample)
            return
        if isinstance(result, str):
            text = result.strip()
            if not text or check_backchannel(text):
                return
            self._start_generation(text, current_sample)
            return
        raise ValueError(f"Unsupported VAD output type: {type(result)}")

    def _request_interruption(self, stop_sample: int):
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        self.stop_event = None
        self.bot_speaking = False
        if hasattr(self.vad, "set_bot_speaking"):
            self.vad.set_bot_speaking(False)

        with self.lock:
            trimmed_segments = []
            for seg in self.assistant_segments:
                audio = seg["audio"]
                if audio.size == 0:
                    continue
                start = seg["start"]
                end = start + len(audio)

                # If segment ended before interruption, keep it
                if end <= stop_sample:
                    trimmed_segments.append(seg)
                    continue

                # If segment starts AFTER interruption, drop it
                if start >= stop_sample:
                    continue

                # Overlap: Trim audio only
                keep_len = stop_sample - start
                if keep_len > 0:
                    seg["audio"] = audio[:keep_len]
                    trimmed_segments.append(seg)

            self.assistant_segments = trimmed_segments
            self.assistant_end_sample = max(
                (seg["start"] + len(seg["audio"]) for seg in self.assistant_segments),
                default=0,
            )

    def _start_generation(self, text: str, current_sample: int):
        # Commit previous turn history if available, with truncation
        self._flush_assistant_history()
        self._request_interruption(current_sample)
        print(f"[OfflineSimulator] Taking Turn. User: {text}")

        flag = threading.Event()
        self.stop_event = flag
        self.llm.add_message(self.client_id, "user", text)

        worker = threading.Thread(
            target=self._generation_worker,
            args=(text, current_sample, flag),
            daemon=True,
        )
        worker.start()
        self.generation_threads.append(worker)

    def _generation_worker(
        self,
        user_text: str,
        current_sample: int,
        stop_event: threading.Event,
    ):
        llm_start = time.time()
        try:
            llm_stream = self.llm.generate_with_history(
                self.client_id, stop_event=stop_event
            )
        except Exception as exc:
            print(f"[OfflineSimulator] LLM error: {exc}")
            if self.stop_event is stop_event:
                self.stop_event = None
            return

        if isinstance(llm_stream, str):
            llm_stream = split_into_segments(llm_stream)

        if self.tts is not None:
            self._speak_segments(llm_stream, current_sample, llm_start, stop_event)
        else:
            message_to_add = ""
            total_audio_duration = 0.0
            first_segment_start_sample = None
            first_llm_latency = 0.0
            first_tts_latency = 0.0

            for i, segment in enumerate(llm_stream):
                if stop_event.is_set():
                    break

                seg = segment["text"]
                pcm_chunk = segment["wav"]

                audio = decode_pcm_bytes(pcm_chunk, self.sample_rate, self._resamplers)
                if audio.size == 0:
                    continue

                if stop_event.is_set():
                    break

                message_to_add += seg
                latency = time.time() - llm_start
                if not first_llm_latency:
                    first_llm_latency = latency
                print(f"[OfflineSimulator] LLM segment: '{seg}', delay: {latency:.3f}s")

                if not first_tts_latency:
                    first_tts_latency = first_llm_latency
                    print(
                        f"[OfflineSimulator] TTS first packet delay: {first_tts_latency:.3f}s"
                    )

                total_audio_duration += len(audio) / self.sample_rate
                if first_segment_start_sample is None:
                    # Start sample of the entire assistant response
                    first_segment_start_sample = current_sample + int(
                        latency * self.sample_rate
                    )
                self._enqueue_segment(seg, audio, current_sample, latency)

            # Update Session State for next turn
            with self.lock:
                if message_to_add:
                    self.pending_message = message_to_add
                    self.pending_audio_duration = total_audio_duration
                    self.pending_start_sample = first_segment_start_sample
                    self.interruption_sample = None
                if first_tts_latency:
                    self.time_info.append(
                        {
                            "llm_latency": first_llm_latency,
                            "tts_latency": first_tts_latency,
                        }
                    )

        if self.stop_event is stop_event:
            self.stop_event = None

    def _speak_segments(
        self,
        segments: List[str],
        current_sample: int,
        llm_start: float,
        stop_event: threading.Event,
    ):
        message_to_add = ""
        total_audio_duration = 0.0
        first_segment_start_sample = None
        first_llm_latency = 0.0
        first_tts_latency = 0.0

        for seg in segments:
            if stop_event.is_set():
                break
            message_to_add += seg
            if not first_llm_latency:
                first_llm_latency = time.time() - llm_start
            print(
                f"[OfflineSimulator] LLM segment: '{seg}', delay: {time.time() - llm_start:.3f}s"
            )

            try:
                for pcm_chunk in self.tts.synthesize(seg, streaming=True):
                    if stop_event.is_set():
                        break
                    decoded = decode_pcm_bytes(
                        pcm_chunk, self.sample_rate, self._resamplers
                    )
                    if decoded.size == 0:
                        continue

                    if stop_event.is_set():
                        break

                    latency = time.time() - llm_start
                    if not first_tts_latency:
                        first_tts_latency = latency
                        print(
                            f"[OfflineSimulator] TTS first packet delay: {first_tts_latency:.3f}s"
                        )
                    total_audio_duration += len(decoded) / self.sample_rate
                    if first_segment_start_sample is None:
                        # Start sample of the entire assistant response
                        first_segment_start_sample = current_sample + int(
                            latency * self.sample_rate
                        )
                    self._enqueue_segment(seg, decoded, current_sample, latency)

            except Exception as exc:
                print(f"[OfflineSimulator] TTS error: {exc}")
                continue

            if stop_event.is_set():
                break

        # Update Session State for next turn
        with self.lock:
            if message_to_add:
                self.pending_message = message_to_add
                self.pending_audio_duration = total_audio_duration
                self.pending_start_sample = first_segment_start_sample
                self.interruption_sample = None
            if first_tts_latency:
                self.time_info.append(
                    {
                        "llm_latency": first_llm_latency,
                        "tts_latency": first_tts_latency,
                    }
                )

    def _enqueue_segment(
        self,
        text: str,
        audio: np.ndarray,
        anchor_sample: int,
        latency_seconds: float,
    ):
        if audio.size == 0:
            return
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with self.lock:
            available_sample = anchor_sample + int(latency_seconds * self.sample_rate)
            start_sample = max(available_sample, self.assistant_end_sample)
            segment = {
                "start": start_sample,
                "audio": audio,
                "text": text,
                "history_added": False,
            }
            self.assistant_segments.append(segment)
            end_sample = start_sample + len(audio)
            if end_sample > self.assistant_end_sample:
                self.assistant_end_sample = end_sample

    def _wait_for_generators(self):
        for thread in list(self.generation_threads):
            thread.join()
        self.generation_threads.clear()
        self._flush_assistant_history()
        if hasattr(self.llm, "sessions"):
            self.llm.sessions.pop(self.client_id, None)

    def _render_assistant_wave(self) -> np.ndarray:
        with self.lock:
            segments = list(self.assistant_segments)
            # print(f"[OfflineSimulator] Total assistant segments: {len(segments)}")

        user_wave = (
            self.user_audio_with_padding
            if self.user_audio_with_padding is not None
            else np.zeros(0, dtype=np.float32)
        )

        if not segments:
            length = (
                len(self.user_audio_with_padding)
                if self.user_audio_with_padding is not None
                else 0
            )
            return np.zeros(length, dtype=np.float32)

        assistant_wave = np.zeros(len(user_wave), dtype=np.float32)
        for seg in segments:
            audio = seg["audio"]
            if audio.size == 0:
                continue
            start = seg["start"]
            end = start + len(audio)
            if start >= len(user_wave):
                continue
            if end > assistant_wave.shape[0]:
                assistant_wave = np.pad(
                    assistant_wave, (0, end - assistant_wave.shape[0])
                )
            assistant_wave[start:end] += audio

        if len(user_wave) - len(assistant_wave) > 0:
            assistant_mix = np.pad(
                assistant_wave, (0, len(user_wave) - len(assistant_wave))
            )
        else:
            assistant_mix = assistant_wave[: len(user_wave)]
        return assistant_mix

    def _render_dialogue_wave(self, assistant_wave: np.ndarray) -> np.ndarray:
        user_wave = (
            self.user_audio_with_padding
            if self.user_audio_with_padding is not None
            else np.zeros(0, dtype=np.float32)
        )
        assert len(user_wave) == len(assistant_wave)
        return user_wave + assistant_wave

    def _flush_assistant_history(self):
        with self.lock:
            if self.pending_message:
                final_msg = self.pending_message
                if (
                    self.interruption_sample is not None
                    and self.pending_start_sample is not None
                    and self.pending_audio_duration > 0
                ):
                    # Calculate elapsed duration (seconds)
                    elapsed_samples = max(
                        0, self.interruption_sample - self.pending_start_sample
                    )
                    elapsed_seconds = elapsed_samples / self.sample_rate

                    ratio = min(elapsed_seconds / self.pending_audio_duration, 1.0)
                    cutoff = int(len(final_msg) * ratio)
                    final_msg = final_msg[:cutoff]
                    print(
                        f"[OfflineSimulator] Previous turn, ratio: {ratio:.2f}, msg: {final_msg}"
                    )
                else:
                    print(f"[OfflineSimulator] Previous turn completed.")

                if final_msg:
                    self.llm.add_message(self.client_id, "assistant", final_msg)

            # Clear pending state
            self.pending_message = None
            self.pending_audio_duration = 0.0
            self.pending_start_sample = None
            self.interruption_sample = None

    def _write_wav(self, path: str, samples: np.ndarray):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        sf.write(path, samples.astype(np.float32), self.sample_rate)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )

    parser.add_argument(
        "--eval_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Path to the evaluation directory",
    )

    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for input wav files"
    )

    parser.add_argument(
        "--streaming_tts", action="store_true", help="Enable streaming TTS"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="demo",
        choices=["demo"],
        help="demo=QwenLLM_stream+IndexTTS_VLLM (matches live app)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only process the first N wav files",
    )

    parser.add_argument(
        "--llm_api",
        type=str,
        default=os.environ.get("LLM_API_URL", "http://127.0.0.1:6007/chat"),
    )

    parser.add_argument(
        "--tts_api",
        type=str,
        default=os.environ.get("TTS_API_URL", "http://127.0.0.1:6006/tts"),
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose output.wav already exists and is >1KB",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    eval_dirs = args.eval_dirs
    prefix = args.prefix
    wav_files = []

    for eval_dir in eval_dirs:
        if eval_dir and os.path.exists(eval_dir) and os.path.isdir(eval_dir):
            for root, _, files in os.walk(eval_dir):
                for name in files:
                    if name == f"{prefix}input.wav" or name == f"input.wav":
                        wav_files.append(os.path.join(root, name))

    if not wav_files:
        print("No wav files found for evaluation.")
        return

    if args.max_samples:
        wav_files = wav_files[: args.max_samples]

    print(f"Found {len(wav_files)} wav files for evaluation.")

    vad_api = TurnTaking()
    print(f"[OfflineSimulator] backend=demo llm={args.llm_api} tts={args.tts_api}")
    llm_api = QwenLLM_stream(api_url=args.llm_api)
    tts_api = IndexTTS_VLLM(api_url=args.tts_api)

    simulator = OfflineDuplexSimulator(
        vad_api=vad_api, llm_api=llm_api, tts_api=tts_api, config=config
    )

    for wav_path in wav_files:
        out_wav_path = wav_path.replace("input.wav", "output.wav")
        if (
            args.skip_existing
            and os.path.isfile(out_wav_path)
            and os.path.getsize(out_wav_path) > 1024
        ):
            print(f"[OfflineSimulator] skip existing: {out_wav_path}", flush=True)
            continue
        print(f"[OfflineSimulator] Processing: {wav_path}")
        dialogue_wav_path = wav_path.replace("input.wav", "dialogue.wav")
        state_path = wav_path.replace("input.wav", "states.json")
        time_info_path = wav_path.replace("input.wav", "infer_time_info.json")

        for _ in range(3):
            try:
                result = simulator.run(
                    wav_path,
                    out_wav_path,
                    dialogue_wav_path,
                    state_path,
                    time_info_path,
                )
            except Exception as e:
                print(f"[OfflineSimulator] Error processing {wav_path}: {e}")
                result = False
            if result:
                break
        print(
            f"[OfflineSimulator] Saved: {out_wav_path} | {dialogue_wav_path} | {state_path} | {time_info_path}\n\n"
        )


if __name__ == "__main__":
    main()

# python dialogue_system/offline_infer.py --config_path config/config.yaml --eval_dirs path/to/eval_dir1 path/to/eval_dir2 --prefix "" --streaming_tts
