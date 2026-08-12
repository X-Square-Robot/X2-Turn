# vLLM MTP examples

After applying the overlay and exporting `final` to `final_vllm`, run:

```bash
MODEL=/path/to/final_vllm bash integrations/vllm/examples/voxtral_mtp/serve.sh
python integrations/vllm/examples/voxtral_mtp/test_stream_mtp.py --audio sample.wav
```

A healthy custom server emits both `transcription.delta` and `turn.delta`.
