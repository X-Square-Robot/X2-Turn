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

## Demo

仓库提供两个定位互补的浏览器 Demo：

- **[Turn Demo](turn-demo/README.md)**：展示流式 ASR、原始六分类 Turn 时间轴、
  每帧概率及对齐后的 ASR token。它不依赖 LLM 或 TTS，是观察模型行为最快的入口。
- **[全双工对话 Demo](full-duplex-demo/README.md)**：将 X2 Turn 与可选的 LLM、
  TTS 服务组合起来，展示低延迟接话、附和以及语音播放期间的用户打断。

评估模型本身请先用 Turn Demo；验证完整对话系统请用全双工 Demo。

### Turn Demo 视频

视频展示了不依赖 LLM 或 TTS 的实时 ASR、六分类 Turn 时间轴及帧级预测结果。

https://github.com/user-attachments/assets/4040eb7a-4f5b-4e25-8ff4-893caeeb0702

### 全双工对话 Demo 视频

视频展示了一次完整会话中的流式 ASR、话轮状态跟踪、回复生成、语音播放及
用户打断效果。

https://github.com/user-attachments/assets/4d322e97-b1ce-4e2e-ac35-d8089d965565

## 最小可行路径：Turn Demo（约 5 分钟）

如果只想检查 ASR + Turn 状态，可以跳过全双工对话栈。
这条路径**不需要** CosyVoice、LLM，也不需要打补丁的 vLLM。

**环境要求：** Python 3.10+、可运行 4B 模型的 CUDA GPU，以及下载
`Kaiqfu/X2-Turn-4B-0812` 的网络访问。

```bash
# 在 X2-Turn 仓库根目录执行
conda env create -f environments/environment-transformers.yml
conda activate x2-turn

cd turn-demo
MODEL=Kaiqfu/X2-Turn-4B-0812 bash run.sh
```

打开 <http://localhost:7860>，选择 **[built-in] English question**，再点击
**Run scenario**。这样即可用内置合成样本完成一次端到端验证，无需麦克风。

这些包**没有**发布到 PyPI。也可以只用 pip 从本仓库安装：

```bash
# 在 X2-Turn 仓库根目录执行
python -m pip install -e "./voxtral-realtime[transformers]"
python -m pip install -e "./turn-demo"
cd turn-demo && MODEL=Kaiqfu/X2-Turn-4B-0812 bash run.sh
```

完整对话系统请看
[`full-duplex-demo/README.md`](full-duplex-demo/README.md)。
那条路径会额外引入 patched vLLM、对话应用，以及外部 CosyVoice 环境。

## 快速开始：本地 Transformers 推理

这条路径**不会**启动 vLLM。它通过本地 Transformers 封装返回转写文本和
80 毫秒 Turn 帧。若要把 WAV 送进生产环境的 turn 控制器，请用
[`voxtral-realtime/examples/offline_inference.py`](voxtral-realtime/examples/README.md)，
那条路径**需要** patched vLLM。

推荐在仓库根目录使用 Miniforge：

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn
```

也可以从本仓库安装 Transformers extra：

```bash
# 在 X2-Turn 仓库根目录执行
python -m pip install -e "./voxtral-realtime[transformers]"
```

在 `voxtral-realtime/` 下安装同一 extra：

```bash
cd voxtral-realtime
python -m pip install -e ".[transformers]"
```

无需修改 Transformers，也无需设置 `trust_remote_code`。下面的音频路径相对
仓库根目录：

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

把同样的结果写成 JSON：

```bash
python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

仓库内置一条合成的 16 kHz 单声道音频。文本、来源、许可证及可复现的
FFmpeg 生成命令见
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
[`vLLM 集成指南`](voxtral-realtime/integrations/vllm/README.md)，
并在 `voxtral-realtime/` 目录下执行其中的命令。标准 vLLM 不会输出自定义的
`turn.delta` 事件。本地服务默认绑定 `127.0.0.1`。

## 发布边界

每个组件均保留独立的许可证与 Notice，因此可以分别发布。模型权重和模型元数据
统一通过
[Hugging Face 模型仓库](https://huggingface.co/Kaiqfu/X2-Turn-4B-0812)
发布，不进入本源码仓库。

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
