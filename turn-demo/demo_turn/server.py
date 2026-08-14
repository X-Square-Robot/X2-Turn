#!/usr/bin/env python3
"""Turn Demo Web UI (FastAPI) — ASR + raw turn states.

Usage:
  # HF (load MTP locally)
  CUDA_VISIBLE_DEVICES=0 python -m demo_turn.server \\
      --model Kaiqfu/X2-Turn-4B-0812 --port 7860

  # vLLM (connect to a running MTP realtime service)
  python -m demo_turn.server --backend vllm \\
      --vllm-url ws://127.0.0.1:8011/v1/realtime \\
      --vllm-model Kaiqfu/X2-Turn-4B-0812 \\
      --port 7860

Click "Start Online Streaming" to view ASR, turn states, and decisions as you speak.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Body, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo_turn.engine_vllm import (
    DEFAULT_VLLM_URL,
    TurnDemoVLLMEngine,
    resolve_vllm_model,
)
from demo_turn.online_vllm import OnlineVLLMSession
from demo_turn.scenarios import build_scenarios
from demo_turn.viz import frame_table_html, timeline_html

ARGS = None
ENGINE = None  # TurnDemoEngine | TurnDemoVLLMEngine
SCENARIOS = []
INFER_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_LOGO_PATH = (
    Path(__file__).resolve().parents[2]
    / "full-duplex-demo"
    / "dialogue_system"
    / "frontend"
    / "x-square-logo.png"
)


class ScenarioReq(BaseModel):
    key: str


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
            from demo_turn.engine import TurnDemoEngine

            ENGINE = TurnDemoEngine(
                model_dir=ARGS.model,
                device=ARGS.device,
                delay_ms=ARGS.delay_ms,
                turn_label_delay_frames=ARGS.turn_label_delay_frames,
            )
    return ENGINE


def run_one(wav_path: str) -> Dict[str, Any]:
    eng = get_engine()
    # Serialize with online HF decode — one MTP generate at a time.
    with INFER_LOCK:
        pred = eng.infer_file(wav_path)
    turns = [f.turn for f in pred.frames]
    return {
        "asr_text": pred.asr_text,
        "last_turn": turns[-1] if turns else "idle",
        "duration_s": pred.duration_s,
        "n_frames": len(pred.frames),
        "timeline_html": timeline_html(
            turns,
            seconds_per_token=pred.seconds_per_token,
        ),
        "frames_html": frame_table_html(pred.frames),
        "turns": turns,
        "turn_hist": {
            k: sum(1 for f in pred.frames if f.turn == k)
            for k in (
                "idle",
                "noidle",
                "speaking",
                "turn_end",
                "backchannel",
                "uncertain",
            )
            if any(f.turn == k for f in pred.frames)
        },
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>X2 Turn Demo</title>
<style>
  :root { --bg:#f8fafc; --card:#fff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, "PingFang SC", "Noto Sans SC", sans-serif;
         background: #fff;
         color: var(--ink); min-height:100vh; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
  .brand { display:flex;align-items:center;gap:14px;margin-bottom:22px; }
  .brand img { width:88px;height:88px;object-fit:contain;border-radius:16px; }
  .brand-copy { min-width:0; }
  h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }
  .sub { color: var(--muted); line-height: 1.5; }
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
  <div class="brand">
    <img src="/assets/x-square-logo.png" alt="X Square mascot"/>
    <div class="brand-copy">
      <h1>X2 Turn Demo</h1>
      <div class="sub">View ASR text and raw Turn model output in real time, one frame every 80 ms.
      Six states: idle / noidle / speaking / turn_end / backchannel / uncertain.</div>
    </div>
  </div>

  <div class="card">
    <table class="guide">
      <tr><th>Turn state</th><th>Model output meaning</th></tr>
      <tr><td><b>idle / noidle</b></td><td>Silence or detected non-silent activity</td></tr>
      <tr><td><b>speaking</b></td><td>The user is still speaking</td></tr>
      <tr><td><b>turn_end</b></td><td>The model predicts the current turn has ended</td></tr>
      <tr><td><b>backchannel</b></td><td>A brief acknowledgment or response</td></tr>
      <tr><td><b>uncertain</b></td><td>The model cannot determine the state yet</td></tr>
    </table>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>Preset scenario</label>
        <select id="scenario"></select>
      </div>
      <div>
        <button id="btn_scene" onclick="runScenario()">Run Scenario</button>
      </div>
    </div>
    <div class="tip" id="scene_tip"></div>
    <audio id="scenario_playback" controls preload="metadata"
      style="width:100%;margin-top:10px;display:none;"></audio>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>Or upload your own WAV file</label>
        <input id="file" type="file" accept="audio/*,.wav"/>
      </div>
      <div>
        <button class="secondary" id="btn_upload" onclick="runUpload()">Analyze Upload</button>
      </div>
    </div>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:8px;">Microphone · Online Streaming</div>
    <div class="tip" style="margin:0 0 12px;">
      Stream over WebSocket as you speak, with incremental ASR and six-state Turn updates every ~320 ms.
      Requires Chrome and localhost or HTTPS.
    </div>
    <div class="row">
      <div>
        <button id="btn_live" onclick="toggleLive()">Start Online Streaming</button>
      </div>
      <div>
        <div id="live_time" style="font-family:ui-monospace,monospace;font-size:20px;padding-top:6px;">00:00</div>
      </div>
      <div class="chk" style="padding-top:8px;">
        <span id="live_state" style="color:#64748b;">Disconnected</span>
      </div>
    </div>
    <div class="tip" id="live_asr" style="margin-top:10px;font-size:15px;color:#0f172a;"></div>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:8px;">Microphone · Record First, Then Analyze (Offline)</div>
    <div class="row">
      <div>
        <button id="btn_mic" class="secondary" onclick="toggleMic()">Start Recording</button>
      </div>
      <div>
        <div id="mic_time" style="font-family:ui-monospace,monospace;font-size:20px;padding-top:6px;">00:00</div>
      </div>
      <div class="chk" style="padding-top:8px;">
        <span id="mic_state" style="color:#64748b;">Not recording</span>
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
      s ? `${s.tip} · Text: ${s.text}` : '';
    const audio = document.getElementById('scenario_playback');
    if (s) {
      audio.src = `/api/scenario_audio/${encodeURIComponent(s.key)}`;
      audio.style.display = 'block';
    } else {
      audio.removeAttribute('src');
      audio.style.display = 'none';
    }
  };
  sel.onchange();
}

function render(j, extraHtml='') {
  document.getElementById('result').innerHTML = `
    <div class="card">${extraHtml}<b>ASR:</b> ${escapeHtml(j.asr_text || '(empty)')}
      <div class="tip">latest turn: <b>${escapeHtml(j.last_turn || 'idle')}</b></div></div>
    <div class="card"><div style="font-size:13px;color:#64748b;margin-bottom:6px;">
      duration=${j.duration_s?.toFixed?.(2)}s · frames=${j.n_frames} · hist=${JSON.stringify(j.turn_hist||{})}
    </div>${j.timeline_html||''}</div>
    <div class="card"><div style="font-weight:600;margin-bottom:8px;">Frame-Level Text Analysis</div>
      ${j.frames_html||''}</div>
  `;
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

async function runScenario() {
  const key = document.getElementById('scenario').value;
  const btn = document.getElementById('btn_scene');
  if (!key) {
    document.getElementById('status').textContent = 'Please select a scenario first';
    return;
  }
  btn.disabled = true;
  document.getElementById('status').textContent = 'Running inference… (the first request may be slower)';
  try {
    const r = await fetch('/api/run_scenario', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({key})
    });
    const j = await r.json();
    if (!r.ok) {
      const detail = j.detail ? JSON.stringify(j.detail) : (j.error || r.statusText);
      throw new Error(detail);
    }
    if (j.error) throw new Error(j.error);
    const tip = `<div style="margin-bottom:10px;color:#475569;font-size:13px;">
      <b>${j.scenario?.title||''}</b><br>${j.scenario?.tip||''}</div>`;
    render(j, tip);
    document.getElementById('status').textContent = 'Done';
  } catch (e) {
    document.getElementById('status').textContent = 'Failed: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function postAudioBlob(blob, filename) {
  const fd = new FormData();
  fd.append('file', blob, filename);
  document.getElementById('status').textContent = 'Running inference… (analyzing the full recording takes a few seconds)';
  const r = await fetch('/api/run_upload', { method:'POST', body: fd });
  let j = {};
  try { j = await r.json(); } catch (_) { j = {}; }
  if (!r.ok) {
    const detail = j.detail ? JSON.stringify(j.detail) : (j.error || r.statusText);
    throw new Error(detail);
  }
  if (j.error) throw new Error(j.error);
  render(j);
  document.getElementById('status').textContent = 'Done';
  return j;
}

async function runUpload() {
  const f = document.getElementById('file').files[0];
  if (!f) { alert('Please select a WAV file first'); return; }
  const btn = document.getElementById('btn_upload');
  btn.disabled = true;
  try {
    await postAudioBlob(f, f.name || 'upload.wav');
  } catch (e) {
    document.getElementById('status').textContent = 'Failed: ' + e.message;
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
    alert('This browser does not support microphone access. Use Chrome over HTTPS or localhost.');
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
  document.getElementById('btn_mic').textContent = 'Stop and Analyze';
  document.getElementById('btn_mic').style.background = '#dc2626';
  document.getElementById('mic_state').textContent = 'Recording…';
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

  document.getElementById('btn_mic').textContent = 'Start Recording';
  document.getElementById('btn_mic').style.background = '';
  document.getElementById('mic_state').textContent = 'Analyzing…';
  document.getElementById('mic_state').style.color = '#64748b';
  document.getElementById('btn_mic').disabled = true;

  let total = 0;
  for (const c of rec.chunks) total += c.length;
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of rec.chunks) { merged.set(c, off); off += c.length; }
  const pcm16k = downsample(merged, rec.sampleRate, 16000);
  if (pcm16k.length < 16000 * 0.3) {
    document.getElementById('status').textContent = 'Recording is too short (minimum ~0.3 s)';
    document.getElementById('mic_state').textContent = 'Not recording';
    document.getElementById('btn_mic').disabled = false;
    return;
  }
  const blob = encodeWav(pcm16k, 16000);
  const audio = document.getElementById('mic_playback');
  audio.src = URL.createObjectURL(blob);
  audio.style.display = 'block';

  try {
    await postAudioBlob(blob, 'mic.wav');
    document.getElementById('mic_state').textContent = 'Analyzed';
  } catch (e) {
    document.getElementById('status').textContent = 'Failed: ' + e.message;
    document.getElementById('mic_state').textContent = 'Failed';
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
      alert('Could not access the microphone: ' + e.message);
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
  document.getElementById('status').textContent = `${j.kind}: turn=${j.last_turn}`;
}

async function startLive() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This browser does not support microphone access');
  }
  let ws = null;
  let stream = null;
  let ctx = null;
  try {
    document.getElementById('live_state').textContent = 'Requesting microphone…';
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    await ctx.resume();

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
    ws.binaryType = 'arraybuffer';
    document.getElementById('live_state').textContent = 'Connecting…';

    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('WebSocket connection timed out')),
        10000,
      );
      ws.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timer);
        reject(new Error('WebSocket connection failed'));
      };
    });

    const ready = new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('Inference backend did not become ready')),
        10000,
      );
      ws.onmessage = (ev) => {
        try {
          const j = JSON.parse(ev.data);
          if (j.error) {
            document.getElementById('status').textContent = 'Error: ' + j.error;
            return;
          }
          if (j.type === 'ready') {
            clearTimeout(timer);
            resolve();
            return;
          }
          if (j.type === 'update' || j.kind === 'partial' || j.kind === 'final') {
            applyStreamUpdate(j);
          }
        } catch (e) {
          console.warn(e);
        }
      };
    });
    ws.send(JSON.stringify({ type: 'start', commit_ms: 320 }));
    await ready;

    const src = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;
    let sendBuf = [];
    let sendSamples = 0;
    const target = Math.floor(ctx.sampleRate * 0.08);

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
        const i16 = new Int16Array(pcm16k.length);
        for (let i = 0; i < pcm16k.length; i++) {
          const s = Math.max(-1, Math.min(1, pcm16k[i]));
          i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        live.ws.send(i16.buffer);
      }
    };

    live = {
      ws, ctx, processor, stream, src, mute,
      startedAt: Date.now(),
      timer: setInterval(liveClock, 250),
    };
    ws.onclose = () => {
      document.getElementById('live_state').textContent = 'Disconnected';
      document.getElementById('live_state').style.color = '#64748b';
    };
    src.connect(processor);
    processor.connect(mute);
    mute.connect(ctx.destination);
    liveClock();

    document.getElementById('btn_live').textContent = 'Stop Online Streaming';
    document.getElementById('btn_live').style.background = '#dc2626';
    document.getElementById('live_state').textContent = 'Streaming…';
    document.getElementById('live_state').style.color = '#16a34a';
    document.getElementById('status').textContent =
      'Online streaming has started. Please speak…';
  } catch (e) {
    if (ws) {
      try { ws.close(); } catch (_) {}
    }
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (ctx) {
      try { await ctx.close(); } catch (_) {}
    }
    document.getElementById('live_state').textContent = 'Failed';
    document.getElementById('live_state').style.color = '#dc2626';
    throw e;
  }
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
  document.getElementById('btn_live').textContent = 'Start Online Streaming';
  document.getElementById('btn_live').style.background = '';
  document.getElementById('live_state').textContent = 'Finalizing…';
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
  document.getElementById('live_state').textContent = 'Finished';
}

async function toggleLive() {
  if (live) {
    document.getElementById('btn_live').disabled = true;
    try { await stopLive(); }
    finally { document.getElementById('btn_live').disabled = false; }
  } else {
    try { await startLive(); }
    catch (e) { alert('Failed to start online streaming: ' + e.message); }
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

    @app.get("/assets/x-square-logo.png", response_class=FileResponse)
    def brand_logo():
        logo_path = Path(os.environ.get("X2_LOGO_PATH", DEFAULT_LOGO_PATH))
        if not logo_path.is_file():
            return JSONResponse({"error": "brand logo not found"}, status_code=404)
        return FileResponse(logo_path, media_type="image/png")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/api/scenarios")
    def api_scenarios():
        return {
            "scenarios": [
                {
                    "key": s.key,
                    "title": s.title,
                    "category": s.category,
                    "text": s.text,
                    "tip": s.tip,
                }
                for s in SCENARIOS
            ]
        }

    @app.get("/api/scenario_audio/{key}", response_class=FileResponse)
    def api_scenario_audio(key: str):
        scenario = next((item for item in SCENARIOS if item.key == key), None)
        if scenario is None or not os.path.isfile(scenario.wav):
            return JSONResponse({"error": "unknown scenario"}, status_code=404)
        return FileResponse(scenario.wav, media_type="audio/wav")

    @app.post("/api/run_scenario")
    async def api_run_scenario(payload: ScenarioReq = Body(...)):
        key = (payload.key or "").strip()
        if not key:
            return JSONResponse({"error": "missing scenario key"}, status_code=400)
        sc = next((s for s in SCENARIOS if s.key == key), None)
        if sc is None:
            return JSONResponse({"error": f"unknown scenario {key}"}, status_code=404)
        try:
            out = await asyncio.to_thread(run_one, sc.wav)
            out["scenario"] = {
                "key": sc.key,
                "title": sc.title,
                "tip": sc.tip,
                "text": sc.text,
                "audio_url": f"/api/scenario_audio/{sc.key}",
            }
            return out
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/run_upload")
    async def api_run_upload(
        file: UploadFile = File(...),
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
                fd2, wav_path = tempfile.mkstemp(
                    suffix=".wav", prefix="demo_turn_norm_"
                )
                os.close(fd2)
                tmp_paths.append(wav_path)
                sf.write(wav_path, np.asarray(data, dtype=np.float32), int(sr))
                path = wav_path
            except Exception:
                pass
            try:
                return await asyncio.to_thread(run_one, path)
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
                        commit_ms = int(data.get("commit_ms") or 320)
                        if ARGS.backend == "vllm":
                            eng = get_engine()
                            session = OnlineVLLMSession(
                                vllm_url=ARGS.vllm_url,
                                model=eng.model,
                                commit_ms=commit_ms,
                                delay_ms=eng.delay_ms,
                                turn_label_delay_frames=eng.turn_delay,
                            )
                            await session.connect()
                        else:
                            from demo_turn.online import OnlineTurnSession

                            session = OnlineTurnSession(
                                get_engine(),
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
                    pcm = i16.astype(np.float32) / 32768.0
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
    p.add_argument("--ssl-certfile", default="", help="optional TLS certificate")
    p.add_argument("--ssl-keyfile", default="", help="optional TLS private key")
    return p.parse_args()


def main():
    global ARGS
    ARGS = parse_args()
    if bool(ARGS.ssl_certfile) != bool(ARGS.ssl_keyfile):
        raise SystemExit("--ssl-certfile and --ssl-keyfile must be provided together")
    app = create_app()
    uvicorn.run(
        app,
        host=ARGS.host,
        port=ARGS.port,
        log_level="info",
        ssl_certfile=ARGS.ssl_certfile or None,
        ssl_keyfile=ARGS.ssl_keyfile or None,
    )


if __name__ == "__main__":
    main()
