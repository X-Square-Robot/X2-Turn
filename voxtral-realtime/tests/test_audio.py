import numpy as np

from voxtral_realtime.audio import AcousticVoiceGate


def test_gate_threshold_and_hangover():
    gate = AcousticVoiceGate(
        sample_rate=1000, rms_threshold=0.1, peak_threshold=0.5, hangover_ms=100
    )
    assert not gate.update(np.zeros(20, dtype=np.float32))
    assert gate.update(np.full(20, 0.2, dtype=np.float32))
    assert gate.update(np.zeros(50, dtype=np.float32))
    assert not gate.update(np.zeros(50, dtype=np.float32))


def test_peak_can_open_gate_and_reset():
    gate = AcousticVoiceGate(
        sample_rate=1000, rms_threshold=1.0, peak_threshold=0.5, hangover_ms=10
    )
    assert gate.update(np.array([0.6], dtype=np.float32))
    gate.reset()
    assert not gate.update(np.array([], dtype=np.float32))
