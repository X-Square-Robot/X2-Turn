# Miniforge environments

This page lists the three Conda environments. New users should create
`x2-turn` from the repository [Quick start](../README.md#quick-start) and
come here only when adding patched vLLM or the full-duplex stack.

Run all commands from the `X2-Turn` repository root. Miniforge is recommended
because these files use the `conda-forge` channel.

The environment files install local editable packages from this checkout. The
Python packages are not published to PyPI.

The services intentionally use separate environments. Combining Transformers,
patched vLLM, CosyVoice, and the dialogue LLM in one environment makes CUDA and
Torch dependency resolution fragile.

## Local Transformers inference and Turn Demo

This is the default environment for one-file ASR + turn inference (no vLLM)
and the standalone browser demo:

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn

python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

To replay a WAV through the production turn controller instead, start patched
vLLM first and use
[`../voxtral-realtime/examples/README.md`](../voxtral-realtime/examples/README.md).

The environment installs PyTorch from PyPI through the
`voxtral-realtime[transformers]` extra. Verify that the resulting Torch build
matches the NVIDIA driver on the target host.

## Patched vLLM

```bash
conda env create -f environments/environment-vllm.yml
conda activate x2-turn-vllm
```

This environment provides the source-build tools but deliberately does not
install stock vLLM. Follow the
[`vLLM overlay guide`](../voxtral-realtime/integrations/vllm/README.md) from the
`voxtral-realtime/` directory to check out the pinned vLLM commit, apply the
X2 Turn overlay, and install the resulting checkout. The supplied Dockerfile is
the preferred option when host CUDA compatibility is uncertain.

## Full-duplex dialogue

```bash
conda env create -f environments/environment-dialogue.yml
conda activate x2-turn-dialogue
```

This environment contains the web app, Edge-TTS fallback, and dialogue LLM
dependencies. The turn vLLM service should run from `x2-turn-vllm`.

CosyVoice must remain in the environment recommended by the upstream CosyVoice
project. Point `COSY_PY` at that environment's Python executable and `VLLM_PY`
at the patched vLLM environment before running `full-duplex-demo/start_demo.sh`.

Example:

```bash
export COSY_PY=/path/to/cosyvoice-env/bin/python
export VLLM_PY=/path/to/x2-turn-vllm-env/bin/python
export VOXTRAL_VLLM_MODEL=/path/to/X2-Turn-4B-0812-vllm
bash full-duplex-demo/start_demo.sh
```

Local services bind to `127.0.0.1` by default. Set `BIND_HOST=0.0.0.0` only
when another machine must connect.

## Recreating environments

Remove an environment before recreating it after dependency changes:

```bash
conda env remove -n x2-turn
conda env create -f environments/environment-transformers.yml
```

Platform-specific lock files should be generated only after validating the
target CUDA driver and GPU architecture.
