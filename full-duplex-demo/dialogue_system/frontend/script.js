const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const voiceCircle = document.getElementById('voiceCircle');
const circleStatusEl = document.getElementById('circleStatus');
const waveformCanvas = document.getElementById('waveformCanvas');
const ctx = waveformCanvas.getContext('2d');

waveformCanvas.width = waveformCanvas.clientWidth;
waveformCanvas.height = 100;

let audioContext, analyser, dataArray, source, stream, processor;
let animationId = null;
let listening = false;

// --- ASR Audio Context is separate from TTS Audio Context ---
let ttsContext = null;
let ttsNode = null;
let ttsSab = null;
let ttsFloat32Data = null;
let ttsStates = null; // Int32Array [writeIndex, readIndex]
let activeAudioEpoch = 0;
let reportedPlaybackEpoch = 0;
let generationDoneEpoch = 0;
let reportedEndedEpoch = 0;
// Buffer size: capacity for ~120 seconds at 24k
const TTS_BUFFER_SIZE = 24000 * 120;

// ==================== Native WebSocket ====================
// Determine protocol (ws or wss)
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws`;

let socket = new WebSocket(wsUrl);
socket.binaryType = 'arraybuffer';

const WS_CONNECT_TIMEOUT_MS = 15000;
const wsConnectTimer = setTimeout(() => {
  if (socket.readyState !== WebSocket.OPEN && loadingMessage) {
    loadingMessage.textContent = 'Cannot connect to server. Refresh or check network.';
  }
}, WS_CONNECT_TIMEOUT_MS);

socket.onopen = () => {
  clearTimeout(wsConnectTimer);
  if (loadingMessage) loadingMessage.textContent = 'Connected, loading models...';
};

socket.onclose = () => {
  if (loadingMessage && loadingOverlay && !loadingOverlay.classList.contains('hidden')) {
    loadingMessage.textContent = 'Disconnected from server. Please refresh.';
  }
};

socket.onerror = () => {
  if (loadingMessage) loadingMessage.textContent = 'WebSocket error. Please refresh.';
};

// Handle incoming messages
socket.onmessage = (event) => {
  const data = event.data;

  // Check if data is binary (Audio Chunk)
  if (data instanceof ArrayBuffer) {
    handleAudioChunk(data);
    return;
  }

  // Otherwise handle JSON messages
  try {
    const msg = JSON.parse(data);
    const eventName = msg.event;
    const payload = msg.data;

    switch (eventName) {
      case 'connect_ack':
        console.log('Connection confirmed:', payload);
        break;
      case 'text_response':
        console.log('[LLM]', payload.text);
        appendConversationReply(payload);
        break;
      case 'vad_loading':
        handleVadLoading(payload);
        break;
      case 'vad_status':
        if (vadStatusEl) {
          vadStatusEl.textContent = payload.message || payload.state || "Unknown Status";
          vadStatusEl.dataset.state = payload.state || "idle";
        }
        break;
      case 'circle_status':
        updateCircleState(payload.status);
        break;
      case 'user_transcription':
        handleUserTranscription(payload);
        recordConversationAsr(payload);
        break;
      case 'stop_audio':
        console.log('Received stop_audio signal');
        activeAudioEpoch = Math.max(activeAudioEpoch, Number(payload.epoch) || 0);
        if (ttsNode) ttsNode.port.postMessage({ type: 'abort' });
        updateCircleState("LISTENING");
        if (payload.reason === 'barge_in') {
          vizLogDecision('INTERRUPT', 'barge-in');
        }
        break;
      case 'pause_audio':
        if (ttsNode) ttsNode.port.postMessage({ type: 'pause' });
        updateCircleState("LISTENING");
        break;
      case 'resume_audio':
        if (ttsNode) ttsNode.port.postMessage({ type: 'resume' });
        updateCircleState('SPEAKING');
        break;
      case 'audio_generation_done':
        generationDoneEpoch = Math.max(generationDoneEpoch, Number(payload.epoch) || 0);
        if (
          generationDoneEpoch === activeAudioEpoch &&
          ttsStates &&
          Atomics.load(ttsStates, 1) >= Atomics.load(ttsStates, 0)
        ) {
          reportPlaybackEnded();
        }
        break;
      case 'turn_viz':
        vizOnTurn(payload);
        break;
      default:
        console.log('Unknown event:', eventName);
    }
  } catch (e) {
    console.error('Failed to parse message:', e);
  }
};


// DOM elements
const vadStatusEl = document.getElementById("vadStatus");
const userTranscriptionEl = document.getElementById("userTranscription");
const eventLogEl = document.getElementById("eventLog");
const conversationLogEl = document.getElementById("conversationLog");
const clearConversationBtn = document.getElementById("clearConversationBtn");
const conversationTurns = new Map();
let conversationSequence = 0;
let conversationDraft = null;

// Loading Overlay Elements
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMessage = document.getElementById("loadingMessage");
const loadingConfirmBtn = document.getElementById("loadingConfirmBtn");
const loadingSpinner = document.querySelector(".loading-spinner");
const loadingSuccess = document.querySelector(".loading-success");

function handleVadLoading({ state, message }) {
  loadingOverlay.classList.remove("hidden");
  loadingMessage.textContent = message;

  if (state === "loading") {
    loadingConfirmBtn.classList.add("hidden");
    loadingSpinner.classList.remove("hidden");
    loadingSuccess.classList.add("hidden");
  } else if (state === "ready") {
    loadingSpinner.classList.add("hidden");
    loadingSuccess.classList.remove("hidden");
    loadingConfirmBtn.classList.remove("hidden");
  }
}

// Close loading overlay
loadingConfirmBtn.addEventListener("click", async () => {
  loadingOverlay.classList.add("hidden");
  updateCircleState("READY");

  // Initialize TTS Audio Engine
  if (!ttsContext) {
    await initTTSAudioEngine();
  }
});

function handleUserTranscription({ text }) {
  if (!userTranscriptionEl) return;

  // Truncate logic: keep last N chars if too long
  const MAX_LEN = 20;
  let displayText = text || "No transcription yet";
  if (displayText.length > MAX_LEN) {
    let tail = displayText.slice(-MAX_LEN);

    // Ensure we don't cut off in the middle of a word
    const match = tail.match(/^[a-zA-Z]+/);
    if (match) {
      const extraLen = match[0].length;
      tail = displayText.slice(-(MAX_LEN + extraLen));
    }

    displayText = "..." + tail;
  }

  userTranscriptionEl.textContent = displayText;
  userTranscriptionEl.classList.add("updated");
  window.setTimeout(() => userTranscriptionEl.classList.remove("updated"), 400);
}

function ensureConversationTurn({ epoch, turn_id: turnId }) {
  const numericEpoch = Number(epoch);
  if (!conversationLogEl || !Number.isFinite(numericEpoch) || numericEpoch <= 0) {
    return null;
  }

  if (conversationTurns.has(numericEpoch)) {
    return conversationTurns.get(numericEpoch);
  }

  if (conversationDraft) {
    const entry = conversationDraft;
    conversationDraft = null;
    entry.turn.classList.remove('live');
    entry.turn.dataset.epoch = String(numericEpoch);
    entry.turnLabel.textContent = `Turn ${entry.sequence}`;
    entry.turnLabel.title = turnId || `epoch ${numericEpoch}`;
    entry.assistantText.textContent = 'Waiting for response…';
    conversationTurns.set(numericEpoch, entry);
    return entry;
  }

  const entry = createConversationTurn({ numericEpoch, turnId, live: false });
  conversationTurns.set(numericEpoch, entry);
  return entry;
}

function createConversationTurn({ numericEpoch = null, turnId = '', live = false }) {
  const empty = conversationLogEl.querySelector('.conversation-empty');
  if (empty) empty.remove();

  conversationSequence += 1;
  const turn = document.createElement('section');
  turn.className = live ? 'conversation-turn live' : 'conversation-turn';
  if (numericEpoch !== null) turn.dataset.epoch = String(numericEpoch);

  const header = document.createElement('div');
  header.className = 'conversation-turn-header';

  const turnLabel = document.createElement('span');
  turnLabel.textContent = live
    ? `Turn ${conversationSequence} · LIVE`
    : `Turn ${conversationSequence}`;
  turnLabel.title = live ? 'Live ASR preview' : (turnId || `epoch ${numericEpoch}`);

  const time = document.createElement('span');
  time.textContent = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  header.append(turnLabel, time);

  const userRow = createConversationMessage('USER', 'user', '');
  const assistantRow = createConversationMessage(
    'LLM',
    'assistant',
    live ? 'Listening…' : 'Waiting for response…',
  );
  assistantRow.text.classList.add('pending');

  turn.append(header, userRow.row, assistantRow.row);
  conversationLogEl.appendChild(turn);

  const entry = {
    turn,
    turnLabel,
    sequence: conversationSequence,
    userText: userRow.text,
    assistantText: assistantRow.text,
  };
  scrollConversationToLatest();
  return entry;
}

function ensureConversationDraft() {
  if (!conversationLogEl) return null;
  if (!conversationDraft) {
    conversationDraft = createConversationTurn({ live: true });
  }
  return conversationDraft;
}

function createConversationMessage(label, roleClass, initialText) {
  const row = document.createElement('div');
  row.className = `conversation-message ${roleClass}`;

  const role = document.createElement('span');
  role.className = 'conversation-role';
  role.textContent = label;

  const text = document.createElement('div');
  text.className = 'conversation-text';
  text.textContent = initialText;

  row.append(role, text);
  return { row, text };
}

function recordConversationAsr(payload) {
  const numericEpoch = Number(payload.epoch);
  const entry = Number.isFinite(numericEpoch) && numericEpoch > 0
    ? ensureConversationTurn(payload)
    : ensureConversationDraft();
  if (!entry) return;
  entry.userText.textContent = payload.text || '';
  scrollConversationToLatest();
}

function appendConversationReply(payload) {
  const entry = ensureConversationTurn(payload);
  if (!entry) return;
  if (entry.assistantText.classList.contains('pending')) {
    entry.assistantText.classList.remove('pending');
    entry.assistantText.textContent = '';
  }
  entry.assistantText.textContent += payload.text || '';
  scrollConversationToLatest();
}

function scrollConversationToLatest() {
  if (!conversationLogEl) return;
  window.requestAnimationFrame(() => {
    conversationLogEl.scrollTop = conversationLogEl.scrollHeight;
  });
}

if (clearConversationBtn) {
  clearConversationBtn.addEventListener('click', () => {
    conversationTurns.clear();
    conversationDraft = null;
    conversationSequence = 0;
    conversationLogEl.replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'conversation-empty';
    empty.textContent = 'No conversation yet';
    conversationLogEl.appendChild(empty);
  });
}

// ==================== TTS Engine (SharedArrayBuffer + AudioWorklet) ====================
async function initTTSAudioEngine() {
  try {
    // Create context with TTS specific sample rate (24k)
    ttsContext = new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' });

    // Add AudioWorklet Module
    await ttsContext.audioWorklet.addModule('processor.js?v=20260812a');

    // Initialize SharedArrayBuffer
    // Header (8 bytes) + Data (Float32 * SIZE)
    const sabSize = 8 + TTS_BUFFER_SIZE * 4;
    ttsSab = new SharedArrayBuffer(sabSize);

    // Views
    ttsStates = new Int32Array(ttsSab, 0, 2); // [0]: WriteIndex, [1]: ReadIndex
    ttsFloat32Data = new Float32Array(ttsSab, 8);

    // Init Indices
    Atomics.store(ttsStates, 0, 0);
    Atomics.store(ttsStates, 1, 0);

    // Create Worklet Node
    ttsNode = new AudioWorkletNode(ttsContext, 'tts-processor', {
      processorOptions: { sab: ttsSab }
    });

    // Handle messages from Worklet (e.g., playback finished)
    ttsNode.port.onmessage = (e) => {
      if (e.data.type === 'playback_started') {
        if (activeAudioEpoch > reportedPlaybackEpoch) {
          reportedPlaybackEpoch = activeAudioEpoch;
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
              event: 'playback_started',
              data: { epoch: activeAudioEpoch }
            }));
          }
        }
      } else if (e.data.type === 'playback_ended') {
        updateCircleState('LISTENING');
        console.log('[TTS] playback finished');
        if (generationDoneEpoch === activeAudioEpoch) reportPlaybackEnded();
      }
    };

    ttsNode.connect(ttsContext.destination);
    console.log("TTS Audio Engine Initialized");

  } catch (e) {
    console.error("Failed to init TTS Engine:", e);
    alert("Audio Engine initialization failed! Only supported in localhost or https. Check console for details.");
  }
}

function reportPlaybackEnded() {
  if (activeAudioEpoch <= reportedEndedEpoch) return;
  reportedEndedEpoch = activeAudioEpoch;
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      event: 'playback_ended',
      data: { epoch: activeAudioEpoch }
    }));
  }
}

// Handle Receive Audio Chunk (Binary)
function handleAudioChunk(arrayBuffer) {
  if (!ttsSab || !ttsFloat32Data) return;

  // Resume context if suspended
  if (ttsContext.state === 'suspended') {
    ttsContext.resume();
  }

  // Convert incoming Int16 PCM to Float32
  // Assuming backend sends raw Int16 bytes from WAV
  const inputInt16 = new Int16Array(arrayBuffer);
  const inputLen = inputInt16.length;

  // Load current Write Index
  let writeIndex = Atomics.load(ttsStates, 0);

  for (let i = 0; i < inputLen; i++) {
    // Normalize Int16 to Float32 [-1, 1]
    const sample = inputInt16[i] / 32768.0;

    // Circular buffer write
    const bufferIndex = writeIndex % TTS_BUFFER_SIZE;
    ttsFloat32Data[bufferIndex] = sample;

    writeIndex++;
  }

  // Update Write Index atomically
  Atomics.store(ttsStates, 0, writeIndex);

  updateCircleState('SPEAKING');
}

// ==================== Audio Capture ====================
startBtn.addEventListener('click', async () => {
  try {
    listening = true;
    updateCircleState("LISTENING");

    // Ensure TTS Engine is ready
    if (!ttsContext) await initTTSAudioEngine();
    if (ttsContext.state === 'suspended') ttsContext.resume();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1,
      }
    });

    // Separate context for recording to avoid sample rate mess
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

    // Send Audio Configuration
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        event: "config_audio",
        data: { sample_rate: audioContext.sampleRate }
      }));
    }

    // Ensure AudioContext is active (required by some browser policies)
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    dataArray = new Uint8Array(analyser.fftSize);

    // Create ScriptProcessor to send in frames
    processor = audioContext.createScriptProcessor(512, 1, 1);
    source.connect(analyser);
    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = e => {
      if (!listening) return;
      const input = e.inputBuffer.getChannelData(0);
      vizCaptureUserPcm(input);
      // Float32 -> Int16
      const int16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        int16[i] = Math.max(-1, Math.min(1, input[i])) * 0x7fff;
      }

      // Native WebSocket Send (Binary)
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(int16.buffer);
      }
    };

    startBtn.disabled = true;
    stopBtn.disabled = false;
    drawUserWaveform();
    pulseAssistantCircle();

  } catch (err) {
    console.error('Cannot access microphone:', err);
    alert("Cannot access microphone! Please check browser permissions.");
    updateCircleState("READY");
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', () => {
  listening = false;

  // Send Stop signal via JSON
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ event: 'duplex_stop' }));
  }

  // Interrupt TTS
  if (ttsNode) ttsNode.port.postMessage({ type: 'abort' });

  cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  voiceCircle.style.transform = 'scale(1)';
  ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  updateCircleState("READY");
});

// ==================== Draw Waveform ====================
function drawUserWaveform() {
  if (!listening) return;
  animationId = requestAnimationFrame(drawUserWaveform);

  analyser.getByteTimeDomainData(dataArray);
  ctx.fillStyle = '#f1f5f9';
  ctx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#0ea5e9';
  ctx.beginPath();

  const sliceWidth = waveformCanvas.width / dataArray.length;
  let x = 0;
  let lastY = waveformCanvas.height / 2;

  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 128.0;
    const y = lastY + (v * waveformCanvas.height / 2 - lastY) * 0.2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    lastY = y;
    x += sliceWidth;
  }
  ctx.stroke();
}

// ==================== Assistant Circle Animation ====================
let pulseDirection = 1;
function pulseAssistantCircle() {
  if (!listening) return;
  animationId = requestAnimationFrame(pulseAssistantCircle);

  // Only animate scale via JS if NOT in CSS-animated states (like speaking/processing)
  if (voiceCircle.classList.contains('state-speaking') || voiceCircle.classList.contains('state-processing')) {
    return;
  }

  let currentScale = parseFloat(voiceCircle.style.transform.replace('scale(', '').replace(')', '')) || 1;
  if (currentScale >= 1.05) pulseDirection = -1;
  if (currentScale <= 1) pulseDirection = 1;
  currentScale += pulseDirection * 0.002;
  voiceCircle.style.transform = `scale(${currentScale})`;
}

// ==================== Realtime Timeline Viz ====================
// 3 lanes on timelineCanvas; decisions go to HTML log (scheme A)
const tlCanvas = document.getElementById('timelineCanvas');
const tlCtx = tlCanvas ? tlCanvas.getContext('2d') : null;
const VIZ_HZ = 25;
const VIZ_SPAN_S = 30;
const VIZ_N = VIZ_HZ * VIZ_SPAN_S;

const TURN_COLORS = {
  idle: '#475569',
  noidle: '#94a3b8',
  speaking: '#f59e0b',
  turn_end: '#06b6d4',
  backchannel: '#a78bfa',
};

const viz = {
  userMin: new Float32Array(VIZ_N),
  userMax: new Float32Array(VIZ_N),
  ai: new Uint8Array(VIZ_N),
  turn: new Array(VIZ_N).fill('idle'),
  head: 0,
  filled: 0,
  currentTurn: 'idle',
  lastAiWrite: 0,
  lastAiRead: -1,
  pendingUserMin: 0,
  pendingUserMax: 0,
  pendingUserSamples: false,
  timer: null,
};

function vizLogDecision(label, detail = '') {
  if (!eventLogEl) return;
  const empty = eventLogEl.querySelector('.event-log-empty');
  if (empty) empty.remove();

  const row = document.createElement('div');
  const key = label.toLowerCase();
  row.className = `event-log-item event-${key}`;
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  row.innerHTML =
    `<span class="event-time">${time}</span>` +
    `<span class="event-badge">${label}</span>` +
    `<span class="event-detail">${detail || ''}</span>`;
  eventLogEl.prepend(row);

  while (eventLogEl.children.length > 40) {
    eventLogEl.removeChild(eventLogEl.lastChild);
  }
}

function vizOnTurn(payload) {
  if (!payload) return;
  viz.currentTurn = payload.turn_class || 'idle';
  if (payload.event === 'accept') {
    const note = payload.reason || payload.note || payload.turn_class || '';
    vizLogDecision('ACCEPT', note);
  } else if (payload.event === 'reject') {
    vizLogDecision('REJECT', payload.turn_class || 'backchannel');
  }
}

function vizCaptureUserPcm(input) {
  if (!listening || !input || input.length === 0) return;
  let chunkMin = 1;
  let chunkMax = -1;
  for (let i = 0; i < input.length; i++) {
    const v = input[i];
    if (v < chunkMin) chunkMin = v;
    if (v > chunkMax) chunkMax = v;
  }
  if (!viz.pendingUserSamples) {
    viz.pendingUserMin = chunkMin;
    viz.pendingUserMax = chunkMax;
    viz.pendingUserSamples = true;
  } else {
    viz.pendingUserMin = Math.min(viz.pendingUserMin, chunkMin);
    viz.pendingUserMax = Math.max(viz.pendingUserMax, chunkMax);
  }
}

function vizTakeUserRange() {
  if (!listening) {
    viz.pendingUserMin = 0;
    viz.pendingUserMax = 0;
    viz.pendingUserSamples = false;
    return [0, 0];
  }
  const range = viz.pendingUserSamples
    ? [viz.pendingUserMin, viz.pendingUserMax]
    : [0, 0];
  viz.pendingUserMin = 0;
  viz.pendingUserMax = 0;
  viz.pendingUserSamples = false;
  return range;
}

function vizAiPlaying() {
  if (!ttsStates) return 0;
  const w = Atomics.load(ttsStates, 0);
  const r = Atomics.load(ttsStates, 1);
  // playing while the read pointer is advancing and behind the write pointer
  const playing = r < w && r !== viz.lastAiRead ? 1 : 0;
  viz.lastAiRead = r;
  return playing;
}

function vizTick() {
  const slot = viz.head % VIZ_N;
  const [userMin, userMax] = vizTakeUserRange();
  viz.userMin[slot] = userMin;
  viz.userMax[slot] = userMax;
  viz.ai[slot] = vizAiPlaying();
  viz.turn[slot] = viz.currentTurn;
  viz.head++;
  if (viz.filled < VIZ_N) viz.filled++;
  vizDraw();
}

function vizDraw() {
  if (!tlCtx) return;
  const W = tlCanvas.width = tlCanvas.clientWidth;
  const H = tlCanvas.height = tlCanvas.clientHeight || 120;
  tlCtx.clearRect(0, 0, W, H);
  tlCtx.fillStyle = '#ffffff';
  tlCtx.fillRect(0, 0, W, H);

  const laneH = H / 3;
  const px = W / VIZ_N;

  tlCtx.fillStyle = '#e2e8f0';
  for (let i = 1; i < 3; i++) tlCtx.fillRect(0, i * laneH, W, 1);
  tlCtx.fillRect(0, laneH * 0.5, W, 1);
  tlCtx.fillStyle = '#64748b';
  tlCtx.font = '10px ui-sans-serif, system-ui';
  tlCtx.fillText('USER', 4, laneH * 0 + 12);
  tlCtx.fillText('AI', 4, laneH * 1 + 12);
  tlCtx.fillText('TURN', 4, laneH * 2 + 12);

  const n = viz.filled;
  const start = viz.head - n;
  for (let i = 0; i < n; i++) {
    const abs = start + i;
    const slot = ((abs % VIZ_N) + VIZ_N) % VIZ_N;
    const x = W - (n - i) * px;

    const userMin = viz.userMin[slot];
    const userMax = viz.userMax[slot];
    if (userMax - userMin > 0.01) {
      const center = laneH * 0.5;
      const halfHeight = (laneH - 8) * 0.5;
      const top = center - Math.min(1, Math.max(0, userMax * 4)) * halfHeight;
      const bottom = center - Math.max(-1, Math.min(0, userMin * 4)) * halfHeight;
      tlCtx.fillStyle = '#38bdf8';
      tlCtx.fillRect(x, top, Math.max(px, 1), Math.max(1, bottom - top));
    }

    if (viz.ai[slot]) {
      tlCtx.fillStyle = '#34d399';
      tlCtx.fillRect(x, laneH * 1 + 6, Math.max(px, 1), laneH - 12);
    }

    tlCtx.fillStyle = TURN_COLORS[viz.turn[slot]] || '#475569';
    tlCtx.fillRect(x, laneH * 2 + 6, Math.max(px, 1), laneH - 12);
  }
}

function vizStart() {
  if (viz.timer) return;
  viz.timer = setInterval(vizTick, 1000 / VIZ_HZ);
}
function vizStop() {
  if (viz.timer) { clearInterval(viz.timer); viz.timer = null; }
}
vizStart(); // always on; lanes stay empty until data flows

// ==================== Update Circle State ====================
function updateCircleState(state) {
  // Reset classes
  voiceCircle.classList.remove('state-speaking', 'state-listening', 'state-processing');

  let statusText = "READY";

  switch (state) {
    case 'LISTENING':
      voiceCircle.classList.add('state-listening');
      statusText = "LISTENING";
      break;

    case 'THINKING':
      voiceCircle.classList.add('state-processing');
      statusText = "THINKING";
      break;

    case 'SPEAKING':
      voiceCircle.classList.add('state-speaking');
      statusText = "SPEAKING";
      break;

    case 'READY':
    default:
      statusText = "READY";
      break;
  }

  if (circleStatusEl) circleStatusEl.textContent = statusText;
}
