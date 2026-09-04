# Third-party notices

This repository contains code adapted from the
[SoulX-Duplug dialogue-system](https://github.com/Soul-AILab/SoulX-Duplug/tree/dialogue-system).
Consult that project for its copyright and license notices.

Runtime dependencies are installed separately and retain their own licenses:

- `voxtral-realtime` and the Voxtral model selected by `VOXTRAL_MODEL`
- Qwen2.5 (`Qwen/Qwen2.5-3B-Instruct` by default)
- Qwen3TTS-Streaming and its separately installed model package
- PyTorch, Transformers, vLLM, FastAPI, Uvicorn, NumPy, and related packages

Qwen3TTS-Streaming source code and model weights are not distributed in this
repository. Users must review their upstream licenses before use.

Model weights are not covered by this repository's Apache License 2.0 unless
their respective owners state otherwise.
