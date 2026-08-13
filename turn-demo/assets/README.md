# Bundled sample audio

`sample_en.wav` is a 16 kHz, mono, PCM WAV included only for Quickstart and UI
smoke testing.

- Text: `Hello, could you tell me what the weather is like today?`
- Text author: X2 Turn project
- Audio generator: FFmpeg `flite` filter, `kal` voice
- External recording or dataset: none
- License: the repository's Apache-2.0 license

Regenerate the file from the repository root:

```bash
ffmpeg -f lavfi \
  -i "flite=text='Hello, could you tell me what the weather is like today?':voice=kal" \
  -ar 16000 -ac 1 -c:a pcm_s16le \
  turn-demo/assets/sample_en.wav
```

The `flite` filter is used only to generate this test fixture and is not a
runtime dependency.
