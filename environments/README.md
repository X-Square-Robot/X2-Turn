# Miniforge environments

Run all commands in this document from the `x2_turn_opensource` repository
root. Miniforge is recommended because these files use the `conda-forge`
channel.

The services intentionally use separate environments. Combining Transformers,
patched vLLM, CosyVoice, and the dialogue LLM in one environment makes CUDA and
Torch dependency resolution fragile.

## Offline inference and Turn Demo

This is the default environment for one-file ASR + turn inference and the
standalone browser demo:

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn

python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio /path/to/input.wav \
  --output offline_frames.json
```

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
[`vLLM overlay guide`](../voxtral-realtime/integrations/vllm/README.md) to check
out the pinned vLLM commit, apply the X2 Turn overlay, and install the resulting
checkout. The supplied Dockerfile is the preferred option when host CUDA
compatibility is uncertain.

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
bash full-duplex-demo/start_demo.sh
```

## Recreating environments

Remove an environment before recreating it after dependency changes:

```bash
conda env remove -n x2-turn
conda env create -f environments/environment-transformers.yml
```

Platform-specific lock files should be generated only after validating the
target CUDA driver and GPU architecture.
