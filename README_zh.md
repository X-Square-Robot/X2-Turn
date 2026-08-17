<div align="center">
  <h1>
    <img
      src="full-duplex-demo/dialogue_system/frontend/x-square-logo.png"
      alt="X Square 吉祥物"
      width="72"
      align="center"
    >
    X2-Turn
  </h1>
  <p>
    <strong>帧同步流式 ASR 与话轮状态预测</strong>
  </p>
  <p>
    <a href="https://huggingface.co/x-square-robot/X2-Turn-4B-0812"><img src="https://img.shields.io/badge/Hugging%20Face-X2--Turn--4B--0812-yellow" alt="Hugging Face 模型"></a>
    <a href="https://arxiv.org/abs/2608.10878"><img src="https://img.shields.io/badge/arXiv-2608.10878-b31b1b" alt="X2-Turn 论文"></a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="Apache-2.0"></a>
  </p>
</div>

[English](README.md) | [中文](README_zh.md)

## 项目简介

X2 Turn 会转写语音，并每隔 80 毫秒预测一个话轮状态：`idle`、`noidle`、
`speaking`、`turn_end`、`backchannel` 或 `uncertain`。

最快的试用方式是浏览器里的 **Turn Demo**。这条路径需要一块 GPU 和 4B
权重，**不需要** LLM、TTS 或 vLLM。

https://github.com/user-attachments/assets/4040eb7a-4f5b-4e25-8ff4-893caeeb0702

## 快速开始

只走下面这一条路。等 Demo 跑出文末的期望结果后，再看 vLLM、全双工栈和
Python API。

**你需要**

- Linux，一块 NVIDIA GPU，**显存至少 24 GB**
- Miniforge 或 Conda，以及 Git
- 能访问 [Hugging Face](https://huggingface.co/x-square-robot/X2-Turn-4B-0812)
  的网络

**第一次会很慢。** 创建环境、下载 4B 权重、以及第一次点击 **Run scenario**
各自都可能要几分钟。网页可能在模型还没装上 GPU 时就已经打开，这是正常的。

### 1. 创建环境

```bash
git clone https://github.com/X-Square-Robot/X2-Turn.git
cd X2-Turn

conda env create -f environments/environment-transformers.yml
conda activate x2-turn
```

这些包**没有**发布到 PyPI。环境文件会从当前仓库做可编辑安装。

如果已经有带 CUDA 的 PyTorch 环境，也可以只用 pip：

```bash
python -m pip install -e "./voxtral-realtime[transformers]"
python -m pip install -e "./turn-demo"
```

### 2. 下载模型

Demo 会在首次推理时拉取 `x-square-robot/X2-Turn-4B-0812`。如果 Hugging Face
卡住、超时，或访问不了 Hub，就先把权重下载到本地：

```bash
# 在 X2-Turn 仓库根目录，并已 conda activate x2-turn
huggingface-cli download x-square-robot/X2-Turn-4B-0812 \
  --local-dir ./models/X2-Turn-4B-0812
```

下一步把 `MODEL` 指到这个目录。

### 3. 启动 Demo

```bash
cd turn-demo
MODEL=x-square-robot/X2-Turn-4B-0812 bash run.sh
```

如果用了本地下载：

```bash
cd turn-demo
MODEL="$PWD/../models/X2-Turn-4B-0812" bash run.sh
```

等到日志出现 `Uvicorn running on http://127.0.0.1:7860`。服务默认只监听
本机，此时还不会加载 4B 权重。

### 4. 打开页面并运行内置样本

打开 <http://localhost:7860>。

1. 选择 **[built-in] English question**。
2. 点击 **Run scenario**。

不需要麦克风。第一次点击会把模型加载到 GPU，可能要几分钟。
Transformers 可能会打印 `attention_mask` 或 `pad_token` 警告，可以忽略。

### 5. 期望结果

内置音频大约 3.4 秒合成英语。成功时大致如下：

| 字段 | 典型值 |
| --- | --- |
| ASR 文本 | `hello can you tell me what the weather is like today` |
| 帧数 | 约 **53** 帧，每帧 80 毫秒 |
| 直方图 | `idle` 32、`noidle` 4、`speaking` 15、`turn_end` 2 |
| 时间轴 | 先说话，再出现 `turn_end`，然后回到 `idle` |

页面上的提示文本是 *Hello, could you tell me what the weather is like
today?* 上面那行 ASR 是模型输出，不是把提示原文抄下来。不同 GPU 和软件
版本下，计数可能差一两帧。只要转写接近这句话，并且在话音末尾附近看到
`turn_end`，就说明安装成功了。

## Demo 跑通之后

### Python API（本地 Transformers）

不启动服务，也不需要 vLLM。在仓库根目录、已激活 `x2-turn` 时：

```python
import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import (
    infer_asr_turn,
    load_mtp_checkpoint,
)

model_id = "x-square-robot/X2-Turn-4B-0812"  # 或 ./models/X2-Turn-4B-0812
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

把同样的结果写成 JSON：

```bash
python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model x-square-robot/X2-Turn-4B-0812 \
  --audio turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

加载器不会修改 Transformers，也不需要 `trust_remote_code`。细节见
[`voxtral-realtime/integrations/transformers/README.md`](voxtral-realtime/integrations/transformers/README.md)。
内置样本的文本、许可证和 FFmpeg 命令见
[`turn-demo/assets/README.md`](turn-demo/assets/README.md)。

### 全双工对话 Demo

这条路径会接入可选的 LLM 和 TTS，并演示播放过程中的用户打断。它是另一套
安装：patched vLLM、对话应用，通常还要一个外部 CosyVoice 环境。请从
[`full-duplex-demo/README.md`](full-duplex-demo/README.md) 开始。

https://github.com/user-attachments/assets/4d322e97-b1ce-4e2e-ac35-d8089d965565

### 用 vLLM 做实时服务

标准 vLLM 不会输出自定义的 `turn.delta` 事件。请在 `voxtral-realtime/`
目录下按
[`vLLM 集成指南`](voxtral-realtime/integrations/vllm/README.md)
操作。若要把 WAV 送进生产环境的 turn 控制器，等该运行时起来后使用
[`voxtral-realtime/examples/offline_inference.py`](voxtral-realtime/examples/README.md)。

本地服务默认绑定 `127.0.0.1`。只有在需要让其他机器连进来时，才设置
`BIND_HOST=0.0.0.0`。

## 仓库结构

- [`voxtral-realtime/`](voxtral-realtime/README.md)：模型封装、本地 ASR +
  Turn 推理、实时控制器，以及 patched vLLM 集成。
- [`turn-demo/`](turn-demo/README.md)：展示原始 ASR、80 毫秒 Turn 状态和
  帧级 token / class / probability 表的浏览器 Demo。
- [`full-duplex-demo/`](full-duplex-demo/README.md)：带可选 LLM 和 TTS 的
  完整对话栈。
- [`environments/`](environments/README.md)：相互独立的 Miniforge 环境，
  避免 Transformers、patched vLLM 和对话应用挤在同一套 CUDA/Torch 里。

## 发布边界

每个组件均保留独立的许可证与 Notice。模型权重在
[Hugging Face](https://huggingface.co/x-square-robot/X2-Turn-4B-0812)
上发布，不进入本源码仓库。

禁止发布本地日志、证书、数据集、外部源码目录或任何凭据。

## 引用

如果这项工作对你的研究有帮助，请引用：

```bibtex
@article{fu2026x2turn,
  title = {X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction},
  author = {Fu, Kaiqi and Wen, Rime and Lin, Altman and Qin, Shawn and Gan, Roy and Wang, Hao and Wang, Qian},
  journal = {arXiv preprint arXiv:2608.10878},
  year = {2026},
}
```

## 致谢

X2 Turn 建立在开源语音与机器学习社区的模型、研究和基础设施之上。感谢：

- [Mistral AI](https://mistral.ai/) 发布
  [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)，
  为本项目提供实时语音基础模型。
- [SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug) 的语义话轮研究，
  以及全双工 Demo 所适配的对话系统基础。
- [vLLM](https://github.com/vllm-project/vllm) 提供高吞吐推理框架，
  X2 Turn 在此基础上实现实时 overlay。
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
  提供模型加载、音频处理及本地推理生态。
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 提供全双工 Demo
  使用的可选流式 TTS 集成。

更详细的归属与许可证信息，请查看各组件中的 `NOTICE` 以及已有的
`THIRD_PARTY_NOTICES.md` 文档。
