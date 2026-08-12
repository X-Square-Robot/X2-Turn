# X Square Demo 状态轮转逻辑（基于 Voxtral 80ms Turn 流）

模型：`x-square/voxtral-mtp-turn-v3-delay0-zhen`
帧周期：**80ms / 状态**（`audio_length_per_tok=8`, 16kHz, hop=160 → 8×160/16000 = 0.08s）

当前 demo 栈：

```
Browser (:8443)
  └─ dialogue_system/app.py
       ├─ Turn ws://127.0.0.1:8000/turn   ← voxtral-realtime
       │         └─ vLLM ws://127.0.0.1:8011/v1/realtime
       ├─ LLM  http://127.0.0.1:6007/chat
       └─ TTS  http://127.0.0.1:6017/tts
```

---

## 1. 模型输出是什么粒度？

### 1.1 推理：80ms 一帧、一状态

Voxtral MTP turn 头在流式推理时，与 ASR 共用 **80ms 延迟流** 时间轴。  
vLLM realtime 每个 `turn.delta` 带 `frame_index`，表示**第几个 80ms 帧**的状态，六类之一：

| ID | 类名 | 含义（训练语义） |
|----|------|------------------|
| 35 | `idle` | 无有效话轮语义；**帧级填充**，字间/句间/静音 |
| 36 | `noidle` | 该 80ms 窗内有语音（粗声学） |
| 37 | `speaking` | 用户正在说、句未结束 |
| 38 | `turn_end` | 可接话/句末 |
| 39 | `backchannel` | 附和，不当完整一轮 |

**不是**「一个字一个状态」。ASR 仍按字/词出 `transcription.delta`，但 turn 是 **parallel 80ms 帧序列**。

### 1.2 训练标签长什么样？

`build_turn_streaming_labels()`（`models/voxtral_asr/data.py`）：

1. 先铺一条长度 = 总生成帧数的数组，**默认全是 `idle`**
2. 再把每个字/词的 turn 类（bc / noidle / speaking / turn_end）**写到 ASR 对齐到的帧区间**
3. 字与字之间、PAD、尾静音 → 保持 **`idle`**

因此真实 gold 时间线类似（示意）：

```
帧:  idle idle idle | bc | idle | noidle | idle | speaking speaking | turn_end | idle idle
      ─── 静音 ───   附和  间隔   有声      间隔    在说            句末       尾静音
```

要点：

- **`idle` 会插在 bc / noidle / speaking / turn_end 中间**，这是正常标签，不是噪声
- 不能用「连续 4 帧无 idle」这种规则——中间一旦出现 idle 就会断计数
- 决策单位应是 **80ms 帧序列 + 逐帧状态机**，不是「最后一个非 idle 字」

---

## 2. 已实现：帧级控制器

实现文件：

| 文件 | 作用 |
|------|------|
| `voxtral_realtime.turn.controller` | N/K 状态机 |
| `voxtral_realtime.server` | WebSocket bridge，逐帧调用控制器 |
| `voxtral_realtime.realtime` | `consume_turn_frames()` 按帧 drain |
| `dialogue_system/clients/vad_client.py` | 每包 audio 附带 `bot_speaking` |
| `dialogue_system/app.py` | TTS 首包设 `bot_speaking=True`，interrupt 清 False |

### 2.1 默认参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `N` (`end_confirm_frames`) | 1 | `turn_end` 后再等 N 帧（×80ms）无恢复说话 → ACCEPT |
| `K` (`silence_end_frames`) | 3 | 已有语义说话后，K 帧非 SPEECH → 软 endpoint |
| `commit_ms` | 80 | Bot 不在播时的轮询/门控节奏 |
| `barge_commit_ms` | 80 | **Bot 在播时**切到 80ms 节奏 + 跳过 lead-in gate |
| `min_asr_chars` | 1 | ACCEPT 前 ASR 至少几个字符 |

配置通过 `voxtral-realtime` 环境变量提供，例如
`VOXTRAL_BARGE_COMMIT_MS`。

### 2.2 规则（逐 80ms 帧）

```python
SPEECH = {noidle, speaking}
BARGE_IMMEDIATE = {speaking, turn_end}   # Bot 在播时
```

**① Bot 在播 TTS（`bot_speaking=True`）**

- 切换到 `barge_commit_ms=80`，**跳过 lead-in 能量门控**，用户音频立刻进 vLLM
- turn 模型输出 `speaking` / `turn_end` → **立刻** `nonidle`（`barge_in`）；`noidle` 不打断
- 仍是 **turn 模型决策**，不是前端 RMS 阈值

**② 附和拒识**

- `backchannel` 且尚未出现语义说话（`speaking`/`turn_end`）→ `idle` + `event=reject`

**③ 硬 endpoint（模型打出 turn_end）**

- 收到 `turn_end` → 进入 pending，计数 **N 帧**
- pending 期间若出现 `noidle`/`speaking` → 取消 pending，继续听
- N 帧倒计时结束且 ASR 非空 → **`speak` / ACCEPT**（`reason=turn_end_confirmed`）

**④ 软 endpoint（补 turn_end 缺失）**

- 已进入语义说话段后，连续 **K 帧** 非 SPEECH（`idle` / `backchannel`）
- ASR 非空 → **`speak` / ACCEPT**（`reason=silence_end`）

**⑤ 其它 SPEECH**

- `speaking` / `noidle` → `nonidle`（HOLD，流式 ASR）

### 2.3 映射到 X Square 三态

| 控制器输出 | X Square `state` | App 行为 |
|------------|---------------|----------|
| idle / reject | `idle` | 无动作 |
| nonidle / barge | `nonidle` | `interrupt()` + 流式 ASR |
| speak / accept | `speak` + text | `pipeline_worker` → LLM→TTS |

### 2.4 示例时间线

```
帧:     idle idle | bc | idle | noidle | speaking … | turn_end | idle idle idle idle
X Square:  idle idle | idle(reject) | idle | nonidle | nonidle … | nonidle(pending) | speak
                                              ↑                              ↑ N=4 帧后 ACCEPT
```

若模型未出 `turn_end`：

```
帧:     … speaking speaking | idle idle … (K=8) …
X Square:  … nonidle …         | silence_run 1..8 → speak (silence_end)
```

---

## 3. App 层（L3）

`dialogue_system/app.py` 逻辑：

| Bridge | App |
|--------|-----|
| `nonidle` | `interrupt()` + 流式 ASR 展示 |
| `speak` | `pipeline_worker(text)` |
| `idle` | 无动作 |

**`bot_speaking` 同步：**

1. TTS 首包 `audio_chunk` 发出时：`session.bot_speaking = True`，`vad.set_bot_speaking(True)`
2. 用户抢话 / `interrupt()`：`bot_speaking = False`
3. 每个 mic chunk：`vad.process(chunk, bot_speaking=session.bot_speaking)`

Bridge 也可收 `type=control` + `bot_speaking`（备用）。

---

## 4. 与历史离线 policy 的关系

| | policy.py（离线/可视化） | X Square 帧控制器 |
|--|-------------------------|----------------|
| 输入 | 字→帧展开后的序列 | 原生 `turn.delta` + `frame_index` |
| 打断 | `bot_speaking` 时连续 K 帧 speech | Bot 播时单帧 `speaking`/`turn_end` 即 barge；`noidle` 不打断 |
| 接话 | 末帧 turn 类 | N 帧 confirm turn_end + K 帧 silence 软 end |
| noidle | 参与 barge 计数 | 参与 HOLD，Bot 播时不单独 barge |

历史 policy 仍可用于离线评测；**X Square live 使用独立
`voxtral_realtime.turn.controller`**。

---

## 5. 源码索引

| 文件 | 内容 |
|------|------|
| `voxtral_realtime.realtime` | vLLM 流式 session + `consume_turn_frames()` |
| `voxtral_realtime.turn.controller` | **Live 帧状态机** |
| `voxtral_realtime.server` | X Square WebSocket bridge |
| `dialogue_system/app.py` | L3 抢话 / LLM 管线 / bot_speaking |

---

## 6. 部署备注

- VAD 模型：`x-square/voxtral-mtp-turn-v3-delay0-zhen`
- TTS：CosyVoice2 `:6017`（`TTS_API_URL`）
- UI：`https://localhost:8443`

启动 bridge 示例：

```bash
VOXTRAL_END_CONFIRM_FRAMES=1 \
VOXTRAL_SILENCE_END_FRAMES=3 \
voxtral-realtime serve \
  --model x-square/voxtral-mtp-turn-v3-delay0-zhen \
  --vllm-url ws://127.0.0.1:8011/v1/realtime
```
