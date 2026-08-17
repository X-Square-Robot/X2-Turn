import json
import wave

import numpy as np
import pytest

from voxtral_realtime import offline
from voxtral_realtime.config import RealtimeConfig


def write_wav(path, samples, sample_rate=8000, channels=1):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def test_load_pcm_wav_mixes_and_resamples(tmp_path):
    source = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    path = tmp_path / "input.wav"
    write_wav(path, source, channels=2)

    loaded = offline.load_pcm_wav(path, target_rate=16000)

    assert loaded.dtype == np.float32
    assert loaded.shape == (1600,)
    assert loaded.min() == pytest.approx(-0.5, abs=1e-3)
    assert loaded.max() == pytest.approx(0.5, abs=2e-3)


def test_iter_chunks_pads_final_chunk():
    chunks = list(offline.iter_chunks(np.arange(5, dtype=np.float32), 4))

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(chunks[1], [4, 0, 0, 0])


@pytest.mark.asyncio
async def test_infer_wav_writes_json_outputs(tmp_path, monkeypatch):
    path = tmp_path / "input.wav"
    write_wav(path, np.zeros(800, dtype=np.float32))
    expected_states = [{"chunk_index": 0, "time_ms": 80, "state": "idle"}]
    expected_transcripts = [
        {"time_ms": 80, "text": "hello", "event": "accept", "reason": "test"}
    ]

    async def fake_infer(samples, config, **kwargs):
        assert samples.shape == (1600,)
        assert config.sample_rate == 16000
        return expected_states, expected_transcripts

    monkeypatch.setattr(offline, "infer_samples", fake_infer)
    output_dir = tmp_path / "output"

    states, transcripts = await offline.infer_wav(
        path,
        output_dir=output_dir,
        config=RealtimeConfig(),
    )

    assert states == expected_states
    assert transcripts == expected_transcripts
    assert json.loads((output_dir / "states.json").read_text()) == expected_states
    assert (
        json.loads((output_dir / "transcript.json").read_text()) == expected_transcripts
    )
