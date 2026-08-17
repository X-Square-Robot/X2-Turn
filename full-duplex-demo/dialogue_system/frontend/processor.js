class TTSProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Options passed from Main Thread
        this.sab = options.processorOptions.sab;

        // Memory layout:
        // Int32[0] -> Write Index (Main Thread updates)
        // Int32[1] -> Read Index (Audio Thread updates)
        this.audioStates = new Int32Array(this.sab, 0, 2);

        // Audio Data starts after 8 bytes (2 * 4 bytes)
        this.audioBuffer = new Float32Array(this.sab, 8);

        this.bufferSize = this.audioBuffer.length;
        this.aborted = false;
        this.paused = false;
        this.wasPlaying = false;
        // Tolerate brief TTS underruns between streaming chunks (24kHz).
        this.underrunHangoverSamples = 24000 * 0.5; // 500ms
        this.underrunSamples = 0;

        this.port.onmessage = (e) => {
            if (e.data.type === 'abort') {
                this.handleAbort();
            } else if (e.data.type === 'pause') {
                this.paused = true;
            } else if (e.data.type === 'resume') {
                this.paused = false;
            }
        };
    }

    handleAbort() {
        this.aborted = true;

        // Drain buffer: move read pointer to write pointer
        const writeIndex = Atomics.load(this.audioStates, 0);
        Atomics.store(this.audioStates, 1, writeIndex);

        // // Clear aborted flag after draining to allow new data
        // this.aborted = false;
        // this.wasPlaying = false;
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0]; // Single channel for simplicity

        if (!channel) return true;

        // Handle paused state
        if (this.paused) {
            channel.fill(0); // Silence
            return true;     // Do not advance readIndex
        }

        // Handle abort state
        if (this.aborted) {
            channel.fill(0);

            // Sync read pointer to write pointer (Drain)
            const writeIndex = Atomics.load(this.audioStates, 0);
            Atomics.store(this.audioStates, 1, writeIndex);

            this.aborted = false; // Reset immediately to accept new data
            this.wasPlaying = false;
            return true;
        }

        const currentWriteIndex = Atomics.load(this.audioStates, 0);
        let currentReadIndex = Atomics.load(this.audioStates, 1);

        // Calculate available samples
        let availableSamples = currentWriteIndex - currentReadIndex;

        // Handle overflow / wrap-around or reset logic
        // Since we use increasing indices, direct subtraction works unless Int32 overflows (years to happen at 24k)
        if (availableSamples < 0) {
            // Should not happen in monotonic increasing logic unless reset
            availableSamples = 0;
        }

        // Detect playback end (with hangover so slow TTS chunks don't glitch)
        if (availableSamples > 0) {
            this.underrunSamples = 0;
            if (!this.wasPlaying) {
                this.port.postMessage({ type: 'playback_started' });
            }
            this.wasPlaying = true;
        } else if (this.wasPlaying) {
            this.underrunSamples += channel.length;
            if (this.underrunSamples >= this.underrunHangoverSamples) {
                this.port.postMessage({ type: 'playback_ended' });
                this.wasPlaying = false;
                this.underrunSamples = 0;
            }
        }

        if (availableSamples === 0) {
            // Buffer empty / Underflow
            channel.fill(0);
            return true;
        }

        // Fill output buffer
        for (let i = 0; i < channel.length; i++) {
            if (currentReadIndex < currentWriteIndex) {
                // Read from Circular Buffer
                const bufferIndex = currentReadIndex % this.bufferSize;
                channel[i] = this.audioBuffer[bufferIndex];
                currentReadIndex++;
            } else {
                channel[i] = 0; // Underflow part of the block
            }
        }

        // Update Read Index
        Atomics.store(this.audioStates, 1, currentReadIndex);

        return true;
    }
}

registerProcessor('tts-processor', TTSProcessor);
