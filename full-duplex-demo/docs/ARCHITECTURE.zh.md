# 全双工 Demo 架构

本 Demo 把持续麦克风输入、增量 ASR、话轮判断、流式 LLM、流式 TTS
和用户打断组织在同一条实时链路中。

## 数据流

```text
Browser microphone (16 kHz PCM)
  → dialogue_system/app.py
  → voxtral-realtime /turn bridge
  → patched vLLM /v1/realtime
  → ASR text + 80 ms turn labels
  → Qwen streaming HTTP
  → CosyVoice or Edge-TTS streaming HTTP
  → Browser SharedArrayBuffer + AudioWorklet
```

服务边界：

- `:8011`：带 MTP overlay 的 vLLM realtime 服务。
- `:8000`：`voxtral-realtime` turn bridge。
- `:6007`：可替换的流式 LLM 服务。
- `:6017`：CosyVoice TTS；Edge-TTS fallback 默认使用 `:6016`。
- `:8443`：FastAPI WebSocket 与静态前端。

Demo 只拥有应用编排和适配器。ASR/turn 推理、声学 gate 和帧控制器属于
独立的 `voxtral-realtime` 包；模型和第三方服务权重不随 Demo 分发。

## 会话与生成

每个浏览器 WebSocket 对应一个独立会话，持有：

- turn bridge 客户端和增量 ASR 状态；
- LLM 对话历史；
- 当前 generation epoch 与取消事件；
- 待提交的 assistant 文本和预计播放时长；
- 串行 WebSocket 发送锁。

新话轮开始或有效打断发生时，generation epoch 递增。旧 LLM/TTS 任务即使
稍后返回，也会因为 epoch 失效而被丢弃，避免迟到 PCM 混入新回复。

## LLM 与 TTS 流水线

LLM producer 持续生成文本并按标点和长度切段；TTS consumer 从有界队列
读取文本片段并产生 24 kHz、单声道、16-bit PCM。首个 PCM 发出后会话才
进入 `bot_speaking`，浏览器则通过 AudioWorklet 从共享环形缓冲区播放。

这一边界有两个目的：

1. TTS 不阻塞后续 LLM 文本生成。
2. 每个 PCM chunk 在发送前再次检查 epoch，保证取消安全。

## 用户打断

浏览器显式请求 echo cancellation 和 noise suppression。应用只在 TTS
已经播放且 turn 模型持续判断用户正在说话时清空音频；生成阶段的残留回声
不会提前取消尚未到达的 TTS 首包。

发生有效打断时：

1. 设置当前取消事件并递增 epoch；
2. 向浏览器发送 `stop_audio`，清空播放缓冲；
3. 停止旧 LLM/TTS 消费；
4. 按已播放比例提交 assistant 历史；
5. 接受完整的新用户话轮后启动下一次生成。

详细 turn 标签和 endpoint 规则见
[`STATE_MACHINE.md`](STATE_MACHINE.md)。

## 部署与依赖

`start_demo.sh` 可启动整套服务，也会复用已健康的 vLLM 或 TTS。所有模型
路径、GPU 编号和端口均通过 `.env` 或环境变量提供。生产部署应使用受信任
TLS 证书；自动生成的证书只适合本地开发。

CosyVoice 源码必须作为外部 checkout 安装，不应复制到本仓库。模型权重、
运行日志、证书、录音和评测集同样不属于公开源码边界。

## 当前局限

- ASR 仍可能出现同音词错误、重复识别和句尾漏字。
- endpoint 速度与误切率之间需要按场景调参。
- TTS 首包受后端、GPU 负载和 warm-up 状态影响。
- 浏览器回声消除能力因设备和浏览器实现而异。
- 标准化准确率和延迟应在独立评测仓库中复现，不在 Demo 中固化数据集。
