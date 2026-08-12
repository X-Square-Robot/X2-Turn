from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "vllm/entrypoints/openai/realtime/connection.py",
    "vllm/entrypoints/openai/realtime/protocol.py",
    "vllm/model_executor/models/registry.py",
    "vllm/model_executor/models/voxtral_realtime.py",
    "vllm/outputs.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/__init__.py",
    "vllm/v1/engine/output_processor.py",
    "vllm/v1/outputs.py",
    "vllm/v1/worker/gpu_model_runner.py",
}


def test_patch_has_all_tracked_changes():
    patch = (ROOT / "patches/vllm-voxtral-mtp.patch").read_text()
    paths = {
        line.split(" b/", 1)[1]
        for line in patch.splitlines()
        if line.startswith("diff --git a/")
    }
    assert paths == EXPECTED
    assert "turn.delta" in patch
    assert "vad_lm_head" in patch


def test_overlay_new_files_and_pin_are_present():
    assert (
        ROOT / "integrations/vllm/vllm/model_executor/models/voxtral_mtp_utils.py"
    ).is_file()
    assert (ROOT / "integrations/vllm/tools/export_mtp_for_vllm.py").is_file()
    installer = (ROOT / "scripts/install_vllm_overlay.sh").read_text()
    assert "b1388b1fbf5aaef47937fabe98931211684666a6" in installer
    assert 'PINNED_VERSION="0.19.1"' in installer
