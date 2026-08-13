# X2 Turn

[English](#english) | [中文](#中文)

<a id="english"></a>

## English

X2 Turn extends Voxtral Realtime with two synchronized outputs: streaming
automatic speech recognition and one turn-taking prediction every 80 ms.

The turn head predicts `idle`, `noidle`, `speaking`, `turn_end`,
`backchannel`, or `uncertain`. Applications should smooth these frame-level
predictions instead of treating a single frame as an irreversible action.

## Quick start: one audio file

Recommended Miniforge setup, run from the repository root:

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn
```

Or install the local Transformers integration with pip:

```bash
cd voxtral-realtime
python -m pip install -e ".[transformers]"
```

Load the model without modifying Transformers or enabling
`trust_remote_code`, then obtain both the transcript and all aligned turn
frames:

```python
import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import (
    infer_asr_turn,
    load_mtp_checkpoint,
)

model_id = "Kaiqfu/X2-Turn-4B-0812"
processor = AutoProcessor.from_pretrained(model_id)
model = load_mtp_checkpoint(
    model_id,
    device="cuda",
    dtype=torch.bfloat16,
).eval()

result = infer_asr_turn(model, processor, "turn-demo/assets/sample_en.wav")

print("ASR:", result.transcript)
for frame in result.turn_frames:
    print(frame.start_ms, frame.end_ms, frame.label, frame.confidence)
```

The published Hub model is `Kaiqfu/X2-Turn-4B-0812`. For offline or private
deployments, `model_id` can instead be a local `voxtral-mtp-turn/final`
directory.

For a ready-to-run command that also writes JSON:

```bash
python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

The bundled sample is synthetic 16 kHz mono speech. Its text, provenance,
license, and reproducible FFmpeg command are documented in
[`turn-demo/assets/README.md`](turn-demo/assets/README.md).

## Repository layout

- [`voxtral-realtime/`](voxtral-realtime/README.md) contains the model wrapper,
  local ASR + turn inference, the realtime controller, and the patched vLLM
  integration. Start here when integrating the model into another project.
- [`voxtral-mtp-turn/`](voxtral-mtp-turn/README.md) is the Model Hub release
  staging directory: Model Card, configuration, tokenizer metadata, release
  checks, and optionally the approved weights.
- [`turn-demo/`](turn-demo/README.md) is the focused browser demo for testing
  raw ASR, 80 ms Turn states, and the frame-level token/class/probability table
  without an LLM, TTS service, or product decision policy.
- [`full-duplex-demo/`](full-duplex-demo/README.md) is the browser-based
  full-duplex dialogue demo that combines X2 Turn with optional LLM and TTS
  services.

[`environments/`](environments/README.md) provides separate Miniforge
environments for local Transformers inference, patched vLLM, and the
full-duplex dialogue stack. Keeping these environments separate avoids most
Torch and CUDA dependency conflicts.

For realtime serving, follow the
[`vLLM integration guide`](voxtral-realtime/integrations/vllm/README.md).
Stock vLLM does not emit the custom `turn.delta` events.

## Validate

```bash
python scripts/check_release_language.py

cd voxtral-realtime
python -m pip install -e ".[dev,transformers]"
pytest
pytest integrations/transformers/tests
python scripts/check_public_release.py

cd ../voxtral-mtp-turn
python verify_model_repo.py --allow-weights

cd ../turn-demo
python -m pip install -e .
pytest
python scripts/check_public_release.py

cd ../full-duplex-demo
python -m pip install -e ../voxtral-realtime
pytest
python scripts/check_public_release.py
```

## Release boundary

Each component retains its own license and notices so it can be published
separately. The canonical model weight may exist locally, but this source
repository ignores it. Publish weights to the approved Model Hub only after
completing
[`MODEL_RELEASE_CHECKLIST.md`](voxtral-mtp-turn/MODEL_RELEASE_CHECKLIST.md).
Never publish local logs, certificates, datasets, external checkouts, or
credentials.

---

<a id="中文"></a>

## 中文

X2 Turn 在 Voxtral Realtime 基础上提供两个同步输出：流式自动语音识别，
以及每 80 毫秒一次的话轮状态预测。

话轮预测头输出 `idle`、`noidle`、`speaking`、`turn_end`、
`backchannel` 或 `uncertain`。应用侧应对帧级结果进行平滑处理，
不要将单帧预测直接视为不可撤销的动作。

### 快速开始：推理一条音频

推荐在仓库根目录使用 Miniforge 环境：

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn
```

也可以直接安装本地 Transformers 集成：

```bash
cd voxtral-realtime
python -m pip install -e ".[transformers]"
```

无需修改 Transformers，也无需启用 `trust_remote_code`，即可同时获得
ASR 文本和对齐后的全部话轮帧：

```python
import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import (
    infer_asr_turn,
    load_mtp_checkpoint,
)

model_id = "Kaiqfu/X2-Turn-4B-0812"
processor = AutoProcessor.from_pretrained(model_id)
model = load_mtp_checkpoint(
    model_id,
    device="cuda",
    dtype=torch.bfloat16,
).eval()

result = infer_asr_turn(model, processor, "turn-demo/assets/sample_en.wav")

print("ASR:", result.transcript)
for frame in result.turn_frames:
    print(frame.start_ms, frame.end_ms, frame.label, frame.confidence)
```

已发布的 Hub 模型为 `Kaiqfu/X2-Turn-4B-0812`。离线或私有部署时，
`model_id` 也可以指向本地 `voxtral-mtp-turn/final` 目录。

下面的命令可以直接运行推理并写出 JSON：

```bash
python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

仓库内置一条合成的 16 kHz 单声道音频。文本、来源、许可证及可复现的
FFmpeg 生成命令记录在
[`turn-demo/assets/README.md`](turn-demo/assets/README.md)。

### 仓库结构

- [`voxtral-realtime/`](voxtral-realtime/README.md)：模型封装、本地 ASR +
  Turn 推理、实时控制器及 patched vLLM 集成。集成模型时建议从这里开始。
- [`voxtral-mtp-turn/`](voxtral-mtp-turn/README.md)：Model Hub 发布准备目录，
  包括模型卡、配置、Tokenizer 元数据、发布检查以及审核通过后的权重。
- [`turn-demo/`](turn-demo/README.md)：专注展示原始 ASR、80 毫秒 Turn 状态和
  帧级 token/class/probability 的浏览器 Demo，不依赖 LLM、TTS 或产品决策策略。
- [`full-duplex-demo/`](full-duplex-demo/README.md)：将 X2 Turn 与可选 LLM、
  TTS 服务组合起来的浏览器全双工对话 Demo。

[`environments/`](environments/README.md) 提供三个相互独立的 Miniforge 环境，
分别用于本地 Transformers 推理、patched vLLM 和全双工对话栈，从而避免大部分
Torch 与 CUDA 依赖冲突。

实时服务请参考
[`vLLM 集成指南`](voxtral-realtime/integrations/vllm/README.md)。
标准 vLLM 不会输出自定义的 `turn.delta` 事件。

### 验证

```bash
python scripts/check_release_language.py

cd voxtral-realtime
python -m pip install -e ".[dev,transformers]"
pytest
pytest integrations/transformers/tests
python scripts/check_public_release.py

cd ../voxtral-mtp-turn
python verify_model_repo.py --allow-weights

cd ../turn-demo
python -m pip install -e .
pytest
python scripts/check_public_release.py

cd ../full-duplex-demo
python -m pip install -e ../voxtral-realtime
pytest
python scripts/check_public_release.py
```

### 发布边界

每个组件均保留独立的许可证与 Notice，因此可以分别发布。标准模型权重可能存在于
本地，但源码仓库会忽略该文件。只有完成
[`MODEL_RELEASE_CHECKLIST.md`](voxtral-mtp-turn/MODEL_RELEASE_CHECKLIST.md)
后，才能将权重发布到审核通过的 Model Hub。

禁止发布本地日志、证书、数据集、外部源码目录或任何凭据。
