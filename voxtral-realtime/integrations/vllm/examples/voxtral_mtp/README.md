# vLLM MTP examples

After applying the overlay and exporting the Hugging Face checkpoint for vLLM,
run:

```bash
MODEL=/path/to/X2-Turn-4B-0812-vllm \
  bash integrations/vllm/examples/voxtral_mtp/serve.sh
python integrations/vllm/examples/voxtral_mtp/test_stream_mtp.py --audio sample.wav
```

A healthy custom server emits both `transcription.delta` and `turn.delta`.
