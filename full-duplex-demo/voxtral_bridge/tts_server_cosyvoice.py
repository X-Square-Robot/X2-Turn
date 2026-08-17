"""CosyVoice2-0.5B streaming TTS server for the X Square dialogue demo.

Same HTTP surface as tts_server.py (edge-tts):
  GET  /health      -> {"status": "ok"}
  POST /tts         -> whole wav (24 kHz mono s16)
  POST /tts_stream  -> chunked raw PCM s16le mono 24 kHz

The zero-shot speaker prompt is embedded once at startup; per request only the
text tokens are recomputed, so first audio chunk is fast (~0.2-0.4s on GPU).
"""

import argparse
import asyncio
import io
import os
import sys
import threading
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSY_ROOT = os.environ.get("COSY_ROOT")
if not COSY_ROOT:
    raise RuntimeError(
        "COSY_ROOT is required and must point to an external CosyVoice checkout"
    )
if not os.path.isdir(os.path.join(COSY_ROOT, "cosyvoice")):
    raise RuntimeError("COSY_ROOT must contain the CosyVoice cosyvoice/ package")
MATCHA_ROOT = os.path.join(COSY_ROOT, "third_party", "Matcha-TTS")
PLUGIN_ROOT = os.path.join(ROOT, "cosyvoice_vllm_plugin")

# Must be set before importing vLLM so EngineCore subprocesses load the plugin.
os.environ.setdefault("VLLM_PLUGINS", "cosyvoice2_register")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

sys.path.insert(0, COSY_ROOT)
sys.path.insert(0, MATCHA_ROOT)
sys.path.insert(0, PLUGIN_ROOT)
# vLLM V1 engine runs in a spawned subprocess which must also be able to
# import cosyvoice for the lazily-registered model class
os.environ["PYTHONPATH"] = os.pathsep.join(
    [PLUGIN_ROOT, COSY_ROOT, MATCHA_ROOT, os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

app = FastAPI()

MODEL = None            # CosyVoice2 instance
BASE_INPUT = None       # cached prompt-side model inputs
GEN_LOCK = threading.Lock()
SAMPLE_RATE = 24000


SPK_ID = "demo_speaker"


def _load_wav_soundfile(path, target_sr, min_sr=16000):
    # replaces cosyvoice.utils.file_utils.load_wav: new torchaudio.load
    # requires torchcodec + system ffmpeg libs which this box doesn't have
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    speech = torch.from_numpy(data).unsqueeze(0)
    if sr != target_sr:
        assert sr >= min_sr
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)
    return speech


def load_model(model_dir: str, prompt_wav: str, prompt_text: str,
               use_vllm: bool = False, use_jit: bool = False, use_trt: bool = False):
    global MODEL
    import cosyvoice.cli.frontend as cosy_frontend
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cosy_frontend.load_wav = _load_wav_soundfile

    if use_vllm:
        from vllm import ModelRegistry

        # register by string path so the spawned EngineCore subprocess can
        # import the class itself
        ModelRegistry.register_model(
            "CosyVoice2ForCausalLM", "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM"
        )

        # upstream load_vllm hardcodes gpu_memory_utilization=0.2 (16 GB on
        # 80 GB cards) which is far more than the 0.5B LLM needs; shrink it
        import threading as _threading

        import cosyvoice.cli.model as cosy_model
        from cosyvoice.utils.file_utils import export_cosyvoice2_vllm

        gpu_util = float(os.environ.get("COSY_VLLM_GPU_UTIL", "0.1"))

        def _load_vllm_small(self, model_dir):
            export_cosyvoice2_vllm(self.llm, model_dir, self.device)
            from vllm import EngineArgs, LLMEngine

            engine_args = EngineArgs(
                model=model_dir,
                skip_tokenizer_init=True,
                enable_prompt_embeds=True,
                gpu_memory_utilization=gpu_util,
            )
            self.llm.vllm = LLMEngine.from_engine_args(engine_args)
            self.llm.lock = _threading.Lock()
            del self.llm.llm.model.model.layers

        cosy_model.CosyVoice2Model.load_vllm = _load_vllm_small

    t0 = time.time()
    MODEL = CosyVoice2(model_dir, load_jit=use_jit, load_trt=use_trt, load_vllm=use_vllm, fp16=True)
    MODEL.add_zero_shot_spk(prompt_text, prompt_wav, SPK_ID)
    print(f"[CosyTTS] model + speaker prompt ready in {time.time() - t0:.1f}s", flush=True)


def synth_stream(text: str):
    """Blocking generator: yields PCM s16le bytes at 24 kHz."""
    with GEN_LOCK:
        for out in MODEL.inference_zero_shot(
            text, "", "", zero_shot_spk_id=SPK_ID, stream=True, speed=1.0
        ):
            speech = out["tts_speech"].squeeze(0).numpy()
            pcm = (np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16)
            yield pcm.tobytes()


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "cosyvoice2"}


@app.post("/tts_stream")
async def tts_stream(request: Request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return Response(content=b"", status_code=204)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    DONE = object()

    def producer():
        try:
            for chunk in synth_stream(text):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except Exception as e:
            print(f"[CosyTTS] synth error: {e}", flush=True)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(DONE), loop).result()

    threading.Thread(target=producer, daemon=True).start()

    async def gen():
        while True:
            item = await queue.get()
            if item is DONE:
                break
            yield item

    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return Response(content=b"", media_type="audio/wav", status_code=204)

    pcm = b"".join(await asyncio.to_thread(lambda: list(synth_stream(text))))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6017)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("COSY_MODEL", "FunAudioLLM/CosyVoice2-0.5B"),
    )
    parser.add_argument("--prompt-wav", default=os.path.join(COSY_ROOT, "asset", "zero_shot_prompt.wav"))
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--vllm", action="store_true", help="run the speech-token LLM with vLLM")
    parser.add_argument("--jit", action="store_true", help="use JIT flow encoder")
    parser.add_argument("--trt", action="store_true", help="use TensorRT flow decoder estimator")
    args = parser.parse_args()

    load_model(args.model_dir, args.prompt_wav, args.prompt_text,
               use_vllm=args.vllm, use_jit=args.jit, use_trt=args.trt)

    # Warm common short/medium shapes so the first user turn does not pay
    # vLLM/CUDA graph capture and TensorRT shape initialization costs.
    t0 = time.time()
    for warm_text in (
        "你好。",
        "你好，很高兴认识你。",
        "当然可以，我们先讲一个简短的故事。",
    ):
        for _ in synth_stream(warm_text):
            pass
    print(f"[CosyTTS] warm-up done in {time.time() - t0:.1f}s", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
