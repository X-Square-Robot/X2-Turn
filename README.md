# X2 Turn Open-Source Release

Private validation monorepo for three independently publishable components.

| Component | Purpose | Documentation |
| --- | --- | --- |
| `voxtral-realtime` | Realtime ASR/turn package, controller, vLLM MTP overlay, and offline WAV example | [Core README](voxtral-realtime/README.md) |
| `voxtral-mtp-turn` | Model Hub metadata and release checklist for `voxtral-mtp-turn-v3-delay0-zhen` | [Model card](voxtral-mtp-turn/README.md) |
| `full-duplex-demo` | Browser UI with optional Qwen and CosyVoice/Edge-TTS services | [Demo README](full-duplex-demo/README.md) |

## Start here

- For one-file ASR and turn inference, see
  [`voxtral-realtime/examples/README.md`](voxtral-realtime/examples/README.md).
- For the patched vLLM runtime and `final` to `final_vllm` conversion, see
  [`voxtral-realtime/integrations/vllm/README.md`](voxtral-realtime/integrations/vllm/README.md).
- For the interactive speech dialogue stack, see
  [`full-duplex-demo/README.md`](full-duplex-demo/README.md).

## Validate

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

## Release boundary

Each component retains its own license and notices so it can be published
separately. The canonical model weight may exist locally, but this GitLab
repository ignores it. Publish weights to the approved Model Hub only after
completing
[`MODEL_RELEASE_CHECKLIST.md`](voxtral-mtp-turn/MODEL_RELEASE_CHECKLIST.md).
Never publish local logs, certificates, datasets, external checkouts, or
credentials.
