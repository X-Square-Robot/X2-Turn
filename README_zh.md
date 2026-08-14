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
    一个基于 Voxtral 的模型、两个同步输出，每 80 毫秒预测一次话轮状态。
  </p>
  <p>
    <a href="https://huggingface.co/Kaiqfu/X2-Turn-4B-0812"><img src="https://img.shields.io/badge/Hugging%20Face-X2--Turn--4B--0812-yellow" alt="Hugging Face 模型"></a>
    <a href="https://arxiv.org/abs/2608.10878"><img src="https://img.shields.io/badge/arXiv-2608.10878-b31b1b" alt="X2-Turn 论文"></a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="Apache-2.0"></a>
  </p>
</div>

[English](README.md) | [中文](README_zh.md)

## 项目简介

X2 Turn 在 Voxtral Realtime 基础上提供两个同步输出：流式自动语音识别，
以及每 80 毫秒一次的话轮状态预测。

话轮预测头输出 `idle`、`noidle`、`speaking`、`turn_end`、
`backchannel` 或 `uncertain`。应用侧应对帧级结果进行平滑处理，
不要将单帧预测直接视为不可撤销的动作。

## 论文

模型架构、ASR 锚定监督方法和中英文实验详见：
**[X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and
Turn State Prediction](https://arxiv.org/abs/2608.10878)**
（[PDF](https://arxiv.org/pdf/2608.10878)）。

```bibtex
@article{fu2026x2turn,
  title={X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction},
  author={Fu, Kaiqi and Wen, Rime and Lin, Altman and Qin, Shawn and Gan, Roy and Wang, Hao and Wang, Qian},
  journal={arXiv preprint arXiv:2608.10878},
  year={2026}
}
```

## Demo

仓库提供两个定位互补的浏览器 Demo：

- **[Turn Demo](turn-demo/README.md)**：展示流式 ASR、原始六分类 Turn 时间轴、
  每帧概率及对齐后的 ASR token。它不依赖 LLM 或 TTS，是观察模型行为最快的入口。
- **[全双工对话 Demo](full-duplex-demo/README.md)**：将 X2 Turn 与可选的 LLM、
  TTS 服务组合起来，展示低延迟接话、附和以及语音播放期间的用户打断。

如果需要评估模型本身，请先使用 Turn Demo；如果需要验证完整对话系统，
请使用全双工对话 Demo。

### Turn Demo 视频

视频展示了不依赖 LLM 或 TTS 的实时 ASR、六分类 Turn 时间轴及帧级预测结果。

https://github.com/user-attachments/assets/216d00ee-b7bd-4964-b65c-8d0d16340591

### 全双工对话 Demo 视频

视频展示了一次完整会话中的流式 ASR、话轮状态跟踪、回复生成、语音播放及
用户打断效果。

https://github.com/user-attachments/assets/b64d9c74-7961-42cc-b7d0-392e0fba516e

## 快速开始：推理一条音频

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
`model_id` 也可以指向本地 checkpoint 目录。

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

## 仓库结构

- [`voxtral-realtime/`](voxtral-realtime/README.md)：模型封装、本地 ASR +
  Turn 推理、实时控制器及 patched vLLM 集成。集成模型时建议从这里开始。
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

## 发布边界

每个组件均保留独立的许可证与 Notice，因此可以分别发布。模型权重和模型元数据
统一通过
[Hugging Face 模型仓库](https://huggingface.co/Kaiqfu/X2-Turn-4B-0812)
发布，不进入本源码仓库。

禁止发布本地日志、证书、数据集、外部源码目录或任何凭据。

## 贡献与安全

欢迎参与贡献。提交大型改动前请先阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，
安全问题请按照 [`SECURITY.md`](SECURITY.md) 私下报告。

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
