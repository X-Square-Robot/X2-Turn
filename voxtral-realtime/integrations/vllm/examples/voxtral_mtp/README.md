# vLLM MTP examples

Smoke-test patched vLLM after the overlay guide. New users should finish the
repository [Quick start](../../../../../README.md#quick-start) first.

After applying the overlay and exporting the Hugging Face checkpoint for vLLM,
run these commands from the `voxtral-realtime/` directory.

The bundled sample lives at `turn-demo/assets/sample_en.wav` in the repository
root. There is no `sample.wav` in this folder.

```bash
# from voxtral-realtime/
MODEL=/path/to/X2-Turn-4B-0812-vllm \
  bash integrations/vllm/examples/voxtral_mtp/serve.sh
python integrations/vllm/examples/voxtral_mtp/test_stream_mtp.py \
  --audio ../turn-demo/assets/sample_en.wav
```

`serve.sh` binds to `127.0.0.1:8011` by default. A healthy custom server emits
both `transcription.delta` and `turn.delta`.
