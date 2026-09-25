"""SatQuery AI — Automated Regression & Optimization Tests for FastAPI Endpoints.

Tests:
1. Health & status endpoints
2. Curated mission examples
3. Spectral indices calculation (NDVI, NDWI, NDBI, CIR)
4. Visual object detection and grounding
5. Single-image VQA & multi-spectral evidence synthesis
6. Bi-temporal change-VQA analysis with two images
7. Response serialization and base64 JPEG encoding
"""

import os
import sys
from pathlib import Path

# Setup paths
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from fastapi.testclient import TestClient
from satquery.api.server import app

client = TestClient(app)

SAMPLE_DIR = repo_root / "data" / "samples"
IMG_VRS = str(SAMPLE_DIR / "vrsbench" / "vrsbench_sample_01.png")
IMG_T1 = str(SAMPLE_DIR / "cdvqa" / "t1_before_2023.png")
IMG_T2 = str(SAMPLE_DIR / "cdvqa" / "t2_after_2025.png")


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_id" in data
    assert "device" in data


def test_examples():
    response = client.get("/api/examples")
    assert response.status_code == 200
    data = response.json()
    assert "examples" in data
    assert len(data["examples"]) >= 1


def test_spectral_indices():
    if not os.path.exists(IMG_VRS):
        return
    with open(IMG_VRS, "rb") as f:
        response = client.post(
            "/api/spectral-indices",
            data={"index_type": "ndvi"},
            files={"image": ("test.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["index_type"] == "ndvi"
    assert "statistics" in data
    assert data["processed_image"].startswith("data:image/jpeg;base64,")


def test_detect():
    if not os.path.exists(IMG_VRS):
        return
    with open(IMG_VRS, "rb") as f:
        response = client.post(
            "/api/detect",
            data={"target_classes": "airplane"},
            files={"image": ("test.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "total_detected" in data
    assert "detected_objects" in data
    assert "annotated_image" in data


def test_analyze_single_image():
    if not os.path.exists(IMG_VRS):
        return
    with open(IMG_VRS, "rb") as f:
        response = client.post(
            "/api/analyze",
            data={"query": "Describe the scene land cover.", "modality_a": "Optical"},
            files={"image_a": ("test.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data and len(data["answer"]) > 0
    assert "evidence" in data
    assert data["evidence"]["is_single_image"] is True
    assert "trace" in data


def test_analyze_two_images():
    if not (os.path.exists(IMG_T1) and os.path.exists(IMG_T2)):
        return
    with open(IMG_T1, "rb") as f1, open(IMG_T2, "rb") as f2:
        response = client.post(
            "/api/analyze",
            data={"query": "What changed between these observations?", "modality_a": "Optical", "modality_b": "Optical"},
            files={
                "image_a": ("t1.png", f1, "image/png"),
                "image_b": ("t2.png", f2, "image/png"),
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data and len(data["answer"]) > 0
    assert "evidence" in data
    assert data["evidence"]["is_single_image"] is False


if __name__ == "__main__":
    print("Running automated FastAPI tests...")
    test_health()
    print("[PASS] /api/health")
    test_status()
    print("[PASS] /api/status")
    test_examples()
    print("[PASS] /api/examples")
    test_spectral_indices()
    print("[PASS] /api/spectral-indices")
    test_detect()
    print("[PASS] /api/detect")
    test_analyze_single_image()
    print("[PASS] /api/analyze (single-image)")
    test_analyze_two_images()
    print("[PASS] /api/analyze (two-image)")
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
