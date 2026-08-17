# Third-party notices

This repository contains code adapted from the
[SoulX-Duplug dialogue-system](https://github.com/Soul-AILab/SoulX-Duplug/tree/dialogue-system).
Consult that project for its copyright and license notices.

Runtime dependencies are installed separately and retain their own licenses:

- `voxtral-realtime` and the Voxtral model selected by `VOXTRAL_MODEL`
- Qwen2.5 (`Qwen/Qwen2.5-3B-Instruct` by default)
- CosyVoice (`FunAudioLLM/CosyVoice2-0.5B` by default)
- PyTorch, Transformers, vLLM, FastAPI, Uvicorn, NumPy, and related packages

CosyVoice source code and model weights are not distributed in this repository.
Users must review the upstream code and model licenses before use. The removed
evaluation material and previous vendored CosyVoice tree are not part of this
Apache-licensed distribution.

Model weights are not covered by this repository's Apache License 2.0 unless
their respective owners state otherwise.
