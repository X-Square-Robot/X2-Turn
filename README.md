# X2 Turn Open-Source Release

This private validation repository contains three independently publishable
components:

```text
x2_turn_opensource/
├── voxtral-realtime/   # Core ASR/turn inference package and vLLM MTP overlay
├── voxtral-mtp-turn/   # Model Hub repository and canonical final checkpoint
└── full-duplex-demo/   # Optional LLM, TTS, and browser dialogue demo
```

## Repository boundaries

- `voxtral-realtime` is the reusable core. It owns the realtime vLLM client,
  acoustic gate, turn controller, `/turn` WebSocket service, weight exporter,
  and the pinned vLLM MTP patch.
- `voxtral-mtp-turn` is the model repository for
  `x-square/voxtral-mtp-turn-v3-delay0-zhen`. Its source artifact is the
  canonical `final/model.safetensors`; serving with vLLM requires conversion
  to `final_vllm` using the exporter in `voxtral-realtime`.
- `full-duplex-demo` depends on `voxtral-realtime` and adds the browser UI,
  Qwen LLM adapter, and CosyVoice/Edge-TTS adapters. CosyVoice source and all
  third-party model weights remain external.

## Local validation

```bash
cd voxtral-realtime
python -m pip install -e ".[dev]"
pytest
python scripts/check_public_release.py

cd ../voxtral-mtp-turn
python verify_model_repo.py --allow-weights

cd ../full-duplex-demo
python -m pip install -e ../voxtral-realtime
pytest
python scripts/check_public_release.py
```

## Release order

1. Complete the legal, data-rights, privacy, evaluation, and artifact checks
   in `voxtral-mtp-turn/MODEL_RELEASE_CHECKLIST.md`.
2. Publish and tag `voxtral-realtime` as `v0.1.0`.
3. Upload `voxtral-mtp-turn` with Git LFS to the approved Model Hub namespace.
4. Update dependency URLs and publish `full-duplex-demo`.

Each child directory has its own license, notices, documentation, and release
checks. Do not publish local logs, certificates, external checkouts, datasets,
or credentials.

The canonical model weight may exist in the local checkout, but this GitLab
validation repository intentionally ignores it. Publish weights separately to
the approved Model Hub only after completing the model release checklist.
