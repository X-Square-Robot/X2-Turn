# 基于 Voxtral 的全双工实时语音对话系统

## 系统概述

本 Demo 实现了一个中文全双工实时语音对话系统，支持持续语音输入、增量 ASR、说话轮次判断、用户打断、流式 LLM 生成和流式语音合成。

## 系统架构

```text
Browser Microphone
    → x-square/voxtral-mtp-turn-v3-delay0-zhen ASR/Turn
    → Hybrid Turn Controller
    → Qwen2.5-3B-Instruct
    → CosyVoice2-0.5B
    → Browser AudioWorklet
```

主要组件：

- ASR/Turn：`x-square/voxtral-mtp-turn-v3-delay0-zhen`
- 推理框架：`voxtral-realtime>=0.1.0`
- LLM：`Qwen/Qwen2.5-3B-Instruct`
- TTS：`FunAudioLLM/CosyVoice2-0.5B`
- 服务框架：FastAPI + WebSocket
- 前端播放：SharedArrayBuffer + AudioWorklet

## Turn-Taking

Voxtral 每隔约 80 ms 输出增量 ASR 和 Turn 标签：

- `idle`
- `noidle`
- `speaking`
- `turn_end`
- `backchannel`

系统使用帧级状态机完成轮次判断：

- `commit_ms = 80 ms`
- 结束确认：1 帧
- 静音确认：3 帧
- 普通 ASR tail：2–5 帧
- 短文本 ASR tail：4–7 帧
- Backchannel 确认：2 帧
- Backchannel 兜底：3 帧

短文本使用更长的结束窗口，以降低短暂停顿造成的碎句。

## Turn 与声学 VAD 融合

为了减少拖音被误判为结束的问题，系统融合了 Turn 模型和声学 VAD：

```text
继续说话 = Turn 检测到说话 OR 声学 VAD 检测到声音
确认结束 = Turn 检测到结束 AND 声学 VAD 持续静音
```

当前声学参数：

- RMS 阈值：0.01
- Peak 阈值：0.05
- Hangover：200 ms
- 最大结束否决时间：640 ms

声学 VAD 只负责阻止过早结束，不单独触发用户打断。

## Backchannel

系统只处理句子开头持续出现的 Backchannel：

- 句子内部 Backchannel 不参与结束判断
- AI 播放期间的简单附和不触发回复
- 带有明确意图的 Backchannel 可以触发兜底回复
- “嗯”“哦”“好的”等纯 filler 可以被拒绝

## LLM 与 TTS 流水线

系统使用生产者—消费者架构解耦 LLM 和 TTS：

1. LLM 持续生成增量文本
2. 文本按标点和长度切分
3. 文本片段写入有界队列
4. TTS 独立消费并生成 PCM
5. 浏览器边接收边播放

该设计使 TTS 不会阻塞后续 LLM 文本生成。

## 用户打断

系统使用 Generation Epoch 管理每轮生成。

用户打断 AI 时：

1. 当前 Epoch 失效
2. 浏览器停止播放
3. 取消旧 LLM/TTS 请求
4. 丢弃迟到音频
5. 根据播放进度提交历史
6. 启动新一轮生成

WebSocket 音频采用串行发送，避免 PCM 乱序和播放杂音。

## 前端功能

前端支持：

- 实时 VAD 状态
- 用户 ASR 展示
- User/AI/Turn 时间线
- ACCEPT、REJECT、INTERRUPT 决策日志
- 按轮次展示 ASR 和 LLM 回复
- 浏览器实际播放延迟统计

## 初步延迟

最近一次交互测试结果：

- ACCEPT 到播放 P50：约 940 ms
- 最快：约 739 ms
- 最慢：约 1.26 s
- LLM 首段：约 180–430 ms
- TTS 首包：约 450–650 ms
- 浏览器播放准备：约 60–75 ms

考虑端点确认后，用户说完到听见回复通常约为 1.2–1.8 秒。

以上结果来自交互式 Demo，不代表标准化 Benchmark。

## 当前局限

- ASR 存在同音词错误和重复识别
- 语义未完成片段仍可能被提前提交
- LLM 面对错误 ASR 时可能过度推测
- CosyVoice2 首包仍需约 0.5 秒
- 被取消的旧 TTS 请求可能短暂占用生成锁
- 端点准确率与响应速度之间存在权衡

## 系统级评估

可使用 FDB 数据评估：

- ASR CER/WER
- 句尾漏字率
- Turn 检测准确率
- 误切率、漏切率和碎句率
- Backchannel 误响应率
- 打断准确率与停止延迟
- 用户说完到 ACCEPT 的延迟
- ACCEPT 到 LLM、TTS 和播放的 P50/P90/P95

评估时需要按真实时间间隔回放音频，并模拟 AI 播放状态。冷启动与热启动应分别统计。

## 总结

该 Demo 将增量 ASR、语义 Turn-Taking、声学 VAD、流式 LLM、流式 TTS 和用户打断统一到一套实时状态管理框架中。

系统已具备完整的全双工语音交互能力，后续可重点优化 ASR 鲁棒性、语义未完成检测、TTS 请求取消和端到端响应延迟。
