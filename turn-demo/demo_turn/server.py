#!/usr/bin/env python3
"""Turn Demo Web UI (FastAPI) — 打断 / 拒识 / 接话 / Online Streaming。

用法:
  # HF（本机加载 MTP）
  CUDA_VISIBLE_DEVICES=0 python -m demo_turn.server \\
      --model Kaiqfu/X2-Turn-4B-0812 --port 7860

  # vLLM（对接已启动的 MTP realtime 服务）
  python -m demo_turn.server --backend vllm \\
      --vllm-url ws://127.0.0.1:8011/v1/realtime \\
      --vllm-model Kaiqfu/X2-Turn-4B-0812 \\
      --port 7860

页面上点「开始 Online 流式」即可边说边看 ASR / turn / 决策。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Body, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo_turn.engine import TurnDemoEngine
from demo_turn.engine_vllm import (
    DEFAULT_VLLM_URL,
    TurnDemoVLLMEngine,
    resolve_vllm_model,
)
from demo_turn.online import OnlineTurnSession
from demo_turn.online_vllm import OnlineVLLMSession
from demo_turn.policy import PolicyConfig, run_policy_on_frames
from demo_turn.scenarios import build_scenarios
from demo_turn.viz import decision_banner, events_table, timeline_html

ARGS = None
ENGINE = None  # TurnDemoEngine | TurnDemoVLLMEngine
SCENARIOS = []
INFER_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class ScenarioReq(BaseModel):
    key: str
    barge_in_frames: Optional[int] = 4


def get_engine():
    global ENGINE
    assert ARGS is not None
    if ENGINE is None:
        if ARGS.backend == "vllm":
            model = resolve_vllm_model(ARGS.vllm_model or ARGS.model)
            ENGINE = TurnDemoVLLMEngine(
                vllm_url=ARGS.vllm_url,
                model=model,
                delay_ms=ARGS.delay_ms if ARGS.delay_ms is not None else 480,
                turn_label_delay_frames=ARGS.turn_label_delay_frames,
            )
        else:
            ENGINE = TurnDemoEngine(
                model_dir=ARGS.model,
                device=ARGS.device,
                delay_ms=ARGS.delay_ms,
                turn_label_delay_frames=ARGS.turn_label_delay_frames,
            )
    return ENGINE


def run_one(wav_path: str, bot_speaking: bool, barge_in_frames: int) -> Dict[str, Any]:
    eng = get_engine()
    # Serialize with online HF decode — one MTP generate at a time.
    with INFER_LOCK:
        pred = eng.infer_file(wav_path)
    decision = run_policy_on_frames(
        turns=[f.turn for f in pred.frames],
        turn_probs=[f.turn_prob for f in pred.frames],
        asr_tokens=[f.asr for f in pred.frames],
        seconds_per_token=pred.seconds_per_token,
        bot_speaking=bot_speaking,
        cfg=PolicyConfig(barge_in_frames=int(barge_in_frames)),
        asr_text=pred.asr_text,
    )
    return {
        "action": decision.action,
        "last_turn": decision.last_turn,
        "reason": decision.reason,
        "barge_in_at_s": decision.barge_in_at_s,
        "asr_text": decision.asr_text,
        "duration_s": pred.duration_s,
        "n_frames": len(pred.frames),
        "banner_html": decision_banner(decision),
        "timeline_html": timeline_html(
            [f.turn for f in pred.frames],
            seconds_per_token=pred.seconds_per_token,
            barge_in_at_s=decision.barge_in_at_s,
        ),
        "events_html": events_table(decision.events, only_interesting=True),
        "turn_hist": {
            k: sum(1 for f in pred.frames if f.turn == k)
            for k in ("idle", "noidle", "speaking", "turn_end", "backchannel", "uncertain")
            if any(f.turn == k for f in pred.frames)
        },
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>X2 Turn Demo</title>
<style>
  :root { --bg:#f8fafc; --card:#fff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, "PingFang SC", "Noto Sans SC", sans-serif;
         background: linear-gradient(160deg,#eef2ff 0%,#f8fafc 40%,#ecfeff 100%);
         color: var(--ink); min-height:100vh; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
  h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }
  .sub { color: var(--muted); margin-bottom: 22px; line-height: 1.5; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
          padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 8px 24px rgba(15,23,42,.04); }
  label { display:block; font-size:13px; color:var(--muted); margin-bottom:6px; }
  select, input[type=number], input[type=file] { width:100%; padding:10px 12px; border:1px solid var(--line);
           border-radius:10px; font-size:14px; background:#fff; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; }
  .row > * { flex: 1; min-width: 180px; }
  button { cursor:pointer; border:0; border-radius:10px; padding:11px 16px; font-weight:600;
           background:#0f172a; color:#fff; font-size:14px; }
  button:disabled { opacity:.5; cursor:wait; }
  button.secondary { background:#e2e8f0; color:#0f172a; }
  .tip { font-size:13px; color:var(--muted); margin-top:8px; }
  .chk { display:flex; align-items:center; gap:8px; padding-top: 22px; }
  #status { font-size:13px; color:var(--muted); min-height: 1.2em; }
  table.guide { width:100%; border-collapse:collapse; font-size:13px; }
  table.guide td, table.guide th { border-bottom:1px solid var(--line); padding:8px 6px; text-align:left; }
</style>
</head>
<body>
<div class="wrap">
  <h1>X2 Turn Demo</h1>
  <div class="sub">体验 <b>打断</b>（Bot 播 TTS 时抢话）与 <b>拒识/接话</b>（backchannel vs turn_end vs 半句）。
  基于 6 类: idle / noidle / speaking / turn_end / backchannel / uncertain。
  后端: <b id="backend_tag">__BACKEND__</b></div>

  <div class="card">
    <table class="guide">
      <tr><th>决策</th><th>含义</th><th>主要依据</th></tr>
      <tr><td><b>ACCEPT</b></td><td>可以回复</td><td>末态 turn_end</td></tr>
      <tr><td><b>REJECT</b></td><td>拒识（不当一轮）</td><td>末态 backchannel</td></tr>
      <tr><td><b>HOLD</b></td><td>继续听</td><td>speaking / uncertain / noidle</td></tr>
      <tr><td><b>barge-in</b></td><td>停掉 Bot TTS</td><td>TTS 中连续 noidle/speaking</td></tr>
    </table>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>预设剧本</label>
        <select id="scenario"></select>
      </div>
      <div>
        <label>打断阈值（连续 speech 帧 ×80ms）</label>
        <input id="barge_frames" type="number" min="1" max="10" value="4"/>
      </div>
      <div>
        <button id="btn_scene" onclick="runScenario()">运行剧本</button>
      </div>
    </div>
    <div class="tip" id="scene_tip"></div>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>或上传自己的 wav</label>
        <input id="file" type="file" accept="audio/*,.wav"/>
      </div>
      <div class="chk">
        <input id="bot_speaking" type="checkbox"/>
        <label for="bot_speaking" style="margin:0">模拟 Bot 正在播 TTS（测打断）</label>
      </div>
      <div>
        <button class="secondary" id="btn_upload" onclick="runUpload()">分析上传</button>
      </div>
    </div>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:8px;">麦克风 · Online Streaming</div>
    <div class="tip" style="margin:0 0 12px;">
      WebSocket 边说边推：每 ~320ms 增量解码 ASR + 6 类 turn + 决策灯。
      测打断请勾选上面「模拟 Bot TTS」。需 Chrome + localhost/https。
    </div>
    <div class="row">
      <div>
        <button id="btn_live" onclick="toggleLive()">开始 Online 流式</button>
      </div>
      <div>
        <div id="live_time" style="font-family:ui-monospace,monospace;font-size:20px;padding-top:6px;">00:00</div>
      </div>
      <div class="chk" style="padding-top:8px;">
        <span id="live_state" style="color:#64748b;">未连接</span>
      </div>
    </div>
    <div class="tip" id="live_asr" style="margin-top:10px;font-size:15px;color:#0f172a;"></div>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:8px;">麦克风 · 录完整段再分析（offline）</div>
    <div class="row">
      <div>
        <button id="btn_mic" class="secondary" onclick="toggleMic()">开始录音</button>
      </div>
      <div>
        <div id="mic_time" style="font-family:ui-monospace,monospace;font-size:20px;padding-top:6px;">00:00</div>
      </div>
      <div class="chk" style="padding-top:8px;">
        <span id="mic_state" style="color:#64748b;">未录音</span>
      </div>
    </div>
    <audio id="mic_playback" controls style="width:100%;margin-top:12px;display:none;"></audio>
  </div>

  <div id="status"></div>
  <div id="result"></div>
</div>
<script>
let SCENARIOS = [];
let micRec = null;
let live = null;  // {ws, ctx, processor, stream, ...}

async function init() {
  const r = await fetch('/api/scenarios');
  const j = await r.json();
  SCENARIOS = j.scenarios || [];
  const sel = document.getElementById('scenario');
  sel.innerHTML = '';
  SCENARIOS.forEach((s, i) => {
    const o = document.createElement('option');
    o.value = s.key; o.textContent = s.title;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const s = SCENARIOS.find(x => x.key === sel.value);
    document.getElementById('scene_tip').textContent =
      s ? `期望 ${s.expect} · ${s.tip} · 文本: ${s.text}` : '';
  };
  sel.onchange();
}

function render(j, extraHtml='') {
  document.getElementById('result').innerHTML = `
    <div class="card">${extraHtml}${j.banner_html || ''}</div>
    <div class="card"><div style="font-size:13px;color:#64748b;margin-bottom:6px;">
      duration=${j.duration_s?.toFixed?.(2)}s · frames=${j.n_frames} · hist=${JSON.stringify(j.turn_hist||{})}
    </div>${j.timeline_html||''}</div>
    <div class="card">${j.events_html||''}</div>
  `;
}

async function runScenario() {
  const key = document.getElementById('scenario').value;
  const bargeRaw = document.getElementById('barge_frames').value;
  const barge = Number.parseInt(bargeRaw, 10);
  const btn = document.getElementById('btn_scene');
  if (!key) {
    document.getElementById('status').textContent = '请先选择剧本';
    return;
  }
  btn.disabled = true;
  document.getElementById('status').textContent = '推理中…（首句可能较慢）';
  try {
    const r = await fetch('/api/run_scenario', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        key,
        barge_in_frames: Number.isFinite(barge) && barge > 0 ? barge : 4,
      })
    });
    const j = await r.json();
    if (!r.ok) {
      const detail = j.detail ? JSON.stringify(j.detail) : (j.error || r.statusText);
      throw new Error(detail);
    }
    if (j.error) throw new Error(j.error);
    const tip = `<div style="margin-bottom:10px;color:#475569;font-size:13px;">
      <b>${j.scenario?.title||''}</b> · 期望 <code>${j.scenario?.expect||''}</code><br>${j.scenario?.tip||''}</div>`;
    render(j, tip);
    document.getElementById('status').textContent = '完成';
  } catch (e) {
    document.getElementById('status').textContent = '失败: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function postAudioBlob(blob, filename) {
  const barge = document.getElementById('barge_frames').value;
  const bot = document.getElementById('bot_speaking').checked;
  const fd = new FormData();
  fd.append('file', blob, filename);
  fd.append('bot_speaking', bot ? '1' : '0');
  fd.append('barge_in_frames', barge);
  document.getElementById('status').textContent = '推理中…（说完后分析整段，约几秒）';
  const r = await fetch('/api/run_upload', { method:'POST', body: fd });
  let j = {};
  try { j = await r.json(); } catch (_) { j = {}; }
  if (!r.ok) {
    const detail = j.detail ? JSON.stringify(j.detail) : (j.error || r.statusText);
    throw new Error(detail);
  }
  if (j.error) throw new Error(j.error);
  render(j);
  document.getElementById('status').textContent = '完成';
  return j;
}

async function runUpload() {
  const f = document.getElementById('file').files[0];
  if (!f) { alert('请先选择 wav'); return; }
  const btn = document.getElementById('btn_upload');
  btn.disabled = true;
  try {
    await postAudioBlob(f, f.name || 'upload.wav');
  } catch (e) {
    document.getElementById('status').textContent = '失败: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function encodeWav(float32, sampleRate) {
  const n = float32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data');
  v.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++, o += 2) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function downsample(buf, fromRate, toRate) {
  if (fromRate === toRate) return buf;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(buf.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, buf.length - 1);
    const t = x - i0;
    out[i] = buf[i0] * (1 - t) + buf[i1] * t;
  }
  return out;
}

function updateMicClock() {
  if (!micRec) return;
  const sec = Math.floor((Date.now() - micRec.startedAt) / 1000);
  const mm = String(Math.floor(sec / 60)).padStart(2, '0');
  const ss = String(sec % 60).padStart(2, '0');
  document.getElementById('mic_time').textContent = mm + ':' + ss;
}

async function startMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('当前浏览器不支持麦克风。请用 Chrome，并通过 https 或 localhost 打开。');
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(processor);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  processor.connect(mute);
  mute.connect(ctx.destination);

  micRec = {
    ctx, processor, stream, src, mute, chunks,
    startedAt: Date.now(),
    timer: setInterval(updateMicClock, 250),
    sampleRate: ctx.sampleRate,
  };
  document.getElementById('btn_mic').textContent = '停止并分析';
  document.getElementById('btn_mic').style.background = '#dc2626';
  document.getElementById('mic_state').textContent = '录音中…';
  document.getElementById('mic_state').style.color = '#dc2626';
  updateMicClock();
}

async function stopMicAndAnalyze() {
  if (!micRec) return;
  const rec = micRec;
  micRec = null;
  clearInterval(rec.timer);
  try { rec.processor.disconnect(); } catch (_) {}
  try { rec.src.disconnect(); } catch (_) {}
  try { rec.mute.disconnect(); } catch (_) {}
  rec.stream.getTracks().forEach(t => t.stop());
  try { await rec.ctx.close(); } catch (_) {}

  document.getElementById('btn_mic').textContent = '开始录音';
  document.getElementById('btn_mic').style.background = '';
  document.getElementById('mic_state').textContent = '分析中…';
  document.getElementById('mic_state').style.color = '#64748b';
  document.getElementById('btn_mic').disabled = true;

  let total = 0;
  for (const c of rec.chunks) total += c.length;
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of rec.chunks) { merged.set(c, off); off += c.length; }
  const pcm16k = downsample(merged, rec.sampleRate, 16000);
  if (pcm16k.length < 16000 * 0.3) {
    document.getElementById('status').textContent = '录音太短（至少约 0.3s）';
    document.getElementById('mic_state').textContent = '未录音';
    document.getElementById('btn_mic').disabled = false;
    return;
  }
  const blob = encodeWav(pcm16k, 16000);
  const audio = document.getElementById('mic_playback');
  audio.src = URL.createObjectURL(blob);
  audio.style.display = 'block';

  try {
    await postAudioBlob(blob, 'mic.wav');
    document.getElementById('mic_state').textContent = '已分析';
  } catch (e) {
    document.getElementById('status').textContent = '失败: ' + e.message;
    document.getElementById('mic_state').textContent = '失败';
  } finally {
    document.getElementById('btn_mic').disabled = false;
  }
}

async function toggleMic() {
  if (micRec) {
    await stopMicAndAnalyze();
  } else {
    try {
      await startMic();
    } catch (e) {
      alert('无法打开麦克风: ' + e.message);
    }
  }
}

function liveClock() {
  if (!live) return;
  const sec = Math.floor((Date.now() - live.startedAt) / 1000);
  const mm = String(Math.floor(sec / 60)).padStart(2, '0');
  const ss = String(sec % 60).padStart(2, '0');
  document.getElementById('live_time').textContent = mm + ':' + ss;
}

function applyStreamUpdate(j) {
  if (!j) return;
  document.getElementById('live_asr').textContent =
    (j.kind === 'final' ? '[FINAL] ' : '[LIVE] ') + (j.asr_text || '(…)');
  render(j, `<div class="tip">online ${j.kind} · infer ${j.elapsed_infer_ms}ms · frames=${j.n_frames}</div>`);
  document.getElementById('status').textContent =
    `${j.kind}: ${j.action} · ${j.reason}`;
}

async function startLive() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('浏览器不支持麦克风');
    return;
  }
  const bot = document.getElementById('bot_speaking').checked;
  const barge = Number(document.getElementById('barge_frames').value || 4);
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = 'arraybuffer';

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error('WebSocket 连接失败'));
  });

  ws.send(JSON.stringify({
    type: 'start',
    bot_speaking: bot,
    barge_in_frames: barge,
    commit_ms: 320,
  }));

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  let sendBuf = [];
  let sendSamples = 0;
  const target = Math.floor(ctx.sampleRate * 0.08); // ~80ms 发包

  processor.onaudioprocess = (e) => {
    if (!live || live.ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    sendBuf.push(new Float32Array(input));
    sendSamples += input.length;
    if (sendSamples >= target) {
      let total = 0;
      for (const c of sendBuf) total += c.length;
      const merged = new Float32Array(total);
      let off = 0;
      for (const c of sendBuf) { merged.set(c, off); off += c.length; }
      sendBuf = [];
      sendSamples = 0;
      const pcm16k = downsample(merged, ctx.sampleRate, 16000);
      // int16 LE binary
      const i16 = new Int16Array(pcm16k.length);
      for (let i = 0; i < pcm16k.length; i++) {
        const s = Math.max(-1, Math.min(1, pcm16k[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      live.ws.send(i16.buffer);
    }
  };
  src.connect(processor);
  processor.connect(mute);
  mute.connect(ctx.destination);

  live = {
    ws, ctx, processor, stream, src, mute,
    startedAt: Date.now(),
    timer: setInterval(liveClock, 250),
  };
  liveClock();

  ws.onmessage = (ev) => {
    try {
      const j = JSON.parse(ev.data);
      if (j.error) {
        document.getElementById('status').textContent = '错误: ' + j.error;
        return;
      }
      if (j.type === 'ready') {
        document.getElementById('live_state').textContent = '流式中…';
        document.getElementById('live_state').style.color = '#16a34a';
        return;
      }
      if (j.type === 'update' || j.kind === 'partial' || j.kind === 'final') {
        applyStreamUpdate(j);
      }
    } catch (e) {
      console.warn(e);
    }
  };
  ws.onclose = () => {
    document.getElementById('live_state').textContent = '已断开';
    document.getElementById('live_state').style.color = '#64748b';
  };

  document.getElementById('btn_live').textContent = '停止 Online';
  document.getElementById('btn_live').style.background = '#dc2626';
  document.getElementById('live_state').textContent = '连接中…';
  document.getElementById('status').textContent = 'Online streaming 已开始，请说话…';
}

async function stopLive() {
  if (!live) return;
  const L = live;
  live = null;
  clearInterval(L.timer);
  try { L.processor.disconnect(); } catch (_) {}
  try { L.src.disconnect(); } catch (_) {}
  try { L.mute.disconnect(); } catch (_) {}
  L.stream.getTracks().forEach(t => t.stop());
  try { await L.ctx.close(); } catch (_) {}
  document.getElementById('btn_live').textContent = '开始 Online 流式';
  document.getElementById('btn_live').style.background = '';
  document.getElementById('live_state').textContent = '收尾中…';
  if (L.ws.readyState === WebSocket.OPEN) {
    L.ws.send(JSON.stringify({ type: 'stop' }));
    // wait for final briefly
    await new Promise((resolve) => {
      const t = setTimeout(resolve, 60000);
      const prev = L.ws.onmessage;
      L.ws.onmessage = (ev) => {
        if (prev) prev(ev);
        try {
          const j = JSON.parse(ev.data);
          if (j.kind === 'final' || j.type === 'final') {
            clearTimeout(t);
            resolve();
          }
        } catch (_) {}
      };
    });
    try { L.ws.close(); } catch (_) {}
  }
  document.getElementById('live_state').textContent = '结束';
}

async function toggleLive() {
  if (live) {
    document.getElementById('btn_live').disabled = true;
    try { await stopLive(); }
    finally { document.getElementById('btn_live').disabled = false; }
  } else {
    try { await startLive(); }
    catch (e) { alert('Online 启动失败: ' + e.message); }
  }
}

init();
</script>
</body>
</html>
"""


def create_app() -> FastAPI:
    app = FastAPI(title="Voxtral Turn Demo")

    @app.on_event("startup")
    def _startup():
        global SCENARIOS
        SCENARIOS = build_scenarios(ARGS.test_jsonl, ARGS.preds_jsonl)
        print(f"[demo] {len(SCENARIOS)} scenarios", flush=True)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": ARGS.backend,
            "model_loaded": ENGINE is not None,
        }

    @app.get("/", response_class=HTMLResponse)
    def index():
        tag = "vLLM /v1/realtime" if ARGS.backend == "vllm" else "HF (local GPU)"
        if ARGS.backend == "vllm":
            tag += f" · {ARGS.vllm_url}"
        return INDEX_HTML.replace("__BACKEND__", tag)

    @app.get("/api/scenarios")
    def api_scenarios():
        return {
            "scenarios": [
                {
                    "key": s.key,
                    "title": s.title,
                    "category": s.category,
                    "mode": s.mode,
                    "expect": s.expect,
                    "text": s.text,
                    "tip": s.tip,
                }
                for s in SCENARIOS
            ]
        }

    @app.post("/api/run_scenario")
    async def api_run_scenario(payload: ScenarioReq = Body(...)):
        key = (payload.key or "").strip()
        barge = int(payload.barge_in_frames or 4)
        if not key:
            return JSONResponse({"error": "missing scenario key"}, status_code=400)
        sc = next((s for s in SCENARIOS if s.key == key), None)
        if sc is None:
            return JSONResponse({"error": f"unknown scenario {key}"}, status_code=404)
        try:
            out = await asyncio.to_thread(
                run_one,
                sc.wav,
                sc.mode == "barge_in",
                barge,
            )
            out["scenario"] = {
                "key": sc.key,
                "title": sc.title,
                "expect": sc.expect,
                "tip": sc.tip,
                "text": sc.text,
            }
            return out
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/run_upload")
    async def api_run_upload(
        file: UploadFile = File(...),
        bot_speaking: str = Form("0"),
        barge_in_frames: str = Form("4"),
    ):
        # Must not call blocking GPU / asyncio.run on the event loop thread
        # (breaks vLLM offline + freezes WebSockets). Offload to a worker.
        suffix = os.path.splitext(file.filename or "up.wav")[1] or ".wav"
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": "upload exceeds the 20 MiB limit"},
                status_code=413,
            )
        tmp_paths: List[str] = []
        try:
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="demo_turn_up_")
            os.close(fd)
            tmp_paths.append(path)
            with open(path, "wb") as f:
                f.write(raw)
            # normalize via soundfile when possible (infer_file still resamples to 16k)
            try:
                data, sr = sf.read(path, always_2d=False)
                if getattr(data, "ndim", 1) > 1:
                    data = np.mean(data, axis=-1)
                fd2, wav_path = tempfile.mkstemp(suffix=".wav", prefix="demo_turn_norm_")
                os.close(fd2)
                tmp_paths.append(wav_path)
                sf.write(wav_path, np.asarray(data, dtype=np.float32), int(sr))
                path = wav_path
            except Exception:
                pass
            try:
                return await asyncio.to_thread(
                    run_one,
                    path,
                    bot_speaking in ("1", "true", "True", "yes"),
                    int(barge_in_frames or 4),
                )
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        await ws.accept()
        session = None  # OnlineTurnSession | OnlineVLLMSession
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "text" in msg and msg["text"] is not None:
                    data = json.loads(msg["text"])
                    typ = data.get("type")
                    if typ == "start":
                        bot = bool(data.get("bot_speaking"))
                        barge = int(data.get("barge_in_frames") or 4)
                        commit_ms = int(data.get("commit_ms") or 320)
                        if ARGS.backend == "vllm":
                            eng = get_engine()
                            session = OnlineVLLMSession(
                                vllm_url=ARGS.vllm_url,
                                model=eng.model,
                                bot_speaking=bot,
                                barge_in_frames=barge,
                                commit_ms=commit_ms,
                                delay_ms=eng.delay_ms,
                                turn_label_delay_frames=eng.turn_delay,
                            )
                            await session.connect()
                        else:
                            session = OnlineTurnSession(
                                get_engine(),
                                bot_speaking=bot,
                                barge_in_frames=barge,
                                commit_ms=commit_ms,
                                lock=INFER_LOCK,
                            )
                        await ws.send_json({"type": "ready", "backend": ARGS.backend})
                    elif typ == "stop":
                        if session is None:
                            await ws.send_json({"error": "session not started"})
                            continue
                        if isinstance(session, OnlineVLLMSession):
                            upd = await session.finish()
                        else:
                            upd = await asyncio.to_thread(session.finish)
                        payload = upd.to_dict()
                        payload["type"] = "final"
                        payload["backend"] = ARGS.backend
                        await ws.send_json(payload)
                        break
                    else:
                        await ws.send_json({"error": f"unknown type {typ}"})
                elif "bytes" in msg and msg["bytes"] is not None:
                    if session is None:
                        await ws.send_json({"error": "send start first"})
                        continue
                    raw = msg["bytes"]
                    # int16 LE PCM @ 16kHz mono
                    i16 = np.frombuffer(raw, dtype=np.int16)
                    pcm = (i16.astype(np.float32) / 32768.0)
                    if isinstance(session, OnlineVLLMSession):
                        upd = await session.push_pcm(pcm)
                    else:
                        upd = await asyncio.to_thread(session.push_pcm, pcm)
                    if upd is not None:
                        payload = upd.to_dict()
                        payload["type"] = "update"
                        payload["backend"] = ARGS.backend
                        await ws.send_json(payload)
        except WebSocketDisconnect:
            if isinstance(session, OnlineVLLMSession):
                await session.close()
            return
        except Exception as e:
            if isinstance(session, OnlineVLLMSession):
                await session.close()
            try:
                await ws.send_json({"error": str(e)})
            except Exception:
                pass

    return app


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="Kaiqfu/X2-Turn-4B-0812",
        help="HF checkpoint (backend=hf) or fallback path for vLLM model id",
    )
    p.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="hf = local Transformers MTP; vllm = remote /v1/realtime",
    )
    p.add_argument(
        "--vllm-url",
        default=DEFAULT_VLLM_URL,
        help="vLLM realtime WebSocket URL",
    )
    p.add_argument(
        "--vllm-model",
        default="",
        help="Model id passed in session.update (default: resolve model→*_vllm)",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--delay_ms", type=int, default=None)
    p.add_argument("--turn_label_delay_frames", type=int, default=0)
    p.add_argument(
        "--test_jsonl",
        default="",
        help="optional local scenario JSONL; upload and microphone work without it",
    )
    p.add_argument(
        "--preds_jsonl",
        default="",
        help="optional evaluated predictions JSONL used to select preset scenarios",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main():
    global ARGS
    ARGS = parse_args()
    app = create_app()
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
