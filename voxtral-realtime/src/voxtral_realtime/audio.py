"""Small audio utilities used by the bridge."""

from __future__ import annotations

import numpy as np


class AcousticVoiceGate:
    """Energy-based endpoint veto with a configurable speech hangover."""

    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = 0.010,
        peak_threshold: float = 0.050,
        hangover_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.peak_threshold = peak_threshold
        self.hangover_samples = int(sample_rate * hangover_ms / 1000)
        self.reset()

    def reset(self) -> None:
        self.hangover_left = 0
        self.last_rms = 0.0
        self.last_peak = 0.0

    def update(self, pcm: np.ndarray) -> bool:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return self.hangover_left > 0
        self.last_rms = float(np.sqrt(np.mean(np.square(samples))))
        self.last_peak = float(np.max(np.abs(samples)))
        active = (
            self.last_rms >= self.rms_threshold or self.last_peak >= self.peak_threshold
        )
        if active:
            self.hangover_left = self.hangover_samples
        else:
            self.hangover_left = max(0, self.hangover_left - samples.size)
        return active or self.hangover_left > 0
