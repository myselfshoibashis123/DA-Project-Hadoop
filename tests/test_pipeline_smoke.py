"""Smoke test of the full CV pipeline on a bundled sample image (skipped without weights)."""
from pathlib import Path

import pytest

from computer_vision.detector import weights_available

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "data" / "sample" / "images").glob("*.jpg"))


@pytest.mark.skipif(not SAMPLES, reason="no sample images")
def test_pipeline_on_sample_image(cfg, tmp_path):
    if not weights_available(cfg):
        pytest.skip("model weights not available")
    from computer_vision.pipeline import SafetyPipeline

    pipe = SafetyPipeline(cfg)
    summary = pipe.run_images(SAMPLES[:1], tmp_path, camera_id="cam_test")
    assert summary["frames_analysed"] == 1
    assert (tmp_path / "detections" / "frames.jsonl").exists()
    assert (tmp_path / "detections" / "observations.jsonl").exists()
    assert list((tmp_path / "annotated").glob("*_annotated.jpg"))
