"""AeroLens AI — FastAPI Backend Server for Next.js / React Frontend.

Exposes REST APIs for:
- 5-tool Agentic VLM inference (VQA, Captioning, Grounding, Change-VQA, Optical-SAR Fusion)
- System telemetry and BigEarthNet LoRA domain adaptation status
- Benchmark mission example payloads
"""

from __future__ import annotations

import os
import io
import json
import base64
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# Import agent primitives from app or satquery
import app as agent_module

import threading

app = FastAPI(
    title="AeroLens AI API",
    description="Autonomous Orbital Earth Observation & Agentic Vision-Language Intelligence API",
    version="2.0.0",
)

# Enable CORS for Next.js development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_weights():
    """Warm up Qwen3-VL-4B weights in the background so inference is instant."""
    def _warmup():
        try:
            agent_module._load()
        except Exception as e:
            print(f"[warmup] Background weight loading encountered: {e}")
    threading.Thread(target=_warmup, daemon=True).start()


def pil_to_base64(img: Optional[Image.Image], quality: int = 88) -> Optional[str]:
    """Encode PIL image to base64 data URI.

    Uses JPEG (quality=88) by default — ~60-70% smaller than PNG with
    negligible visual difference for satellite imagery and evidence overlays.
    Next.js/React accepts data:image/jpeg;base64,... equally to PNG.
    """
    if img is None:
        return None
    buf = io.BytesIO()
    rgb_img = img.convert("RGB")  # JPEG does not support alpha
    rgb_img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/api/status")
def get_status():
    """Return spacecraft and inference engine telemetry."""
    is_cuda = agent_module._is_cuda_supported() and os.getenv("SATQUERY_FORCE_CPU", "0") != "1"
    gpu_name = "N/A"
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "status": "online",
        "downlink_freq": "8.2 GHz (X-BAND)",
        "orbit": "LEO 540KM · SSO (98.2°)",
        "model_id": agent_module.MODEL_ID,
        "model_ready": agent_module._model is not None,
        "device": "CUDA GPU" if is_cuda else "CPU Mode (Zero-Crash Fallback)",
        "adaptation": agent_module.adaptation_status(),
        "gpu_name": gpu_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


@app.get("/api/examples")
def get_examples():
    """Return curated benchmark missions for single-scene, change-VQA, and optical-SAR fusion."""
    examples = []
    
    # Check sample files
    try:
        from satquery.core.samples import ensure_sample_imagery
        ensure_sample_imagery()
    except Exception as e:
        print(f"[samples] ensure_sample_imagery notice: {e}")

    candidates = [
        {
            "id": "vrsbench_airfield",
            "title": "Airfield & Infrastructure Grounding",
            "category": "Visual Grounding",
            "mission_tag": "PASS: EO-742",
            "image_a": "data/samples/vrsbench/vrsbench_sample_01.png",
            "modality_a": "Optical",
            "image_b": None,
            "modality_b": "Auto",
            "query": "Locate and highlight all parked airplanes with a bounding box.",
        },
        {
            "id": "rsvqa_landuse",
            "title": "Urban Land-Cover & Buildings",
            "category": "VQA & Counting",
            "mission_tag": "PASS: EO-819",
            "image_a": "data/samples/rsvqa/rsvqa_sample_01.png",
            "modality_a": "Optical",
            "image_b": None,
            "modality_b": "Auto",
            "query": "Is a residential building present in this scene?",
        },
        {
            "id": "cdvqa_flood",
            "title": "Coastal & River Basin Flood Inundation",
            "category": "Bi-Temporal Disaster",
            "mission_tag": "DISASTER CHARTER #581",
            "image_a": "data/samples/disaster/flood_before.png",
            "modality_a": "Optical",
            "image_b": "data/samples/disaster/flood_after.png",
            "modality_b": "Optical",
            "query": "What structural and land-cover changes occurred between these two observation dates?",
        },
        {
            "id": "bigearthnet_fusion",
            "title": "Sentinel-2 Optical + Sentinel-1 SAR Multi-Sensor Fusion",
            "category": "Optical-SAR Fusion",
            "mission_tag": "BEN-DUAL-SENSOR",
            "image_a": "data/samples/bigearthnet/sentinel2_optical.png",
            "modality_a": "Optical",
            "image_b": "data/samples/bigearthnet/sentinel1_sar.png",
            "modality_b": "SAR",
            "query": "Use both the optical and SAR images together to identify water boundaries beneath cloud cover.",
        },
    ]

    for item in candidates:
        if os.path.exists(item["image_a"]):
            # Add thumbnail base64
            try:
                img_a = Image.open(item["image_a"]).convert("RGB")
                img_a.thumbnail((320, 320))
                item["image_a_preview"] = pil_to_base64(img_a)
            except Exception:
                item["image_a_preview"] = None
            
            if item["image_b"] and os.path.exists(item["image_b"]):
                try:
                    img_b = Image.open(item["image_b"]).convert("RGB")
                    img_b.thumbnail((320, 320))
                    item["image_b_preview"] = pil_to_base64(img_b)
                except Exception:
                    item["image_b_preview"] = None
            else:
                item["image_b_preview"] = None
                
            examples.append(item)

    return {"examples": examples}


@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    modality_a: str = Form("Auto"),
    modality_b: str = Form("Auto"),
    image_a: UploadFile = File(...),
    image_b: Optional[UploadFile] = File(None),
):
    """Execute the 5-tool Agentic VLM workflow and return evidence + execution trace."""
    # Save temporary files
    suffix_a = Path(image_a.filename or "image_a.png").suffix or ".png"
    temp_a = tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a)
    temp_a.write(await image_a.read())
    temp_a.close()
    path_a = temp_a.name

    path_b = None
    if image_b is not None:
        suffix_b = Path(image_b.filename or "image_b.png").suffix or ".png"
        temp_b = tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b)
        temp_b.write(await image_b.read())
        temp_b.close()
        path_b = temp_b.name

    try:
        # Run agent
        result = agent_module._agent.run(path_a, path_b, modality_a, modality_b, query)

        if result.error:
            return JSONResponse(
                status_code=400,
                content={"error": result.error, "trace": None},
            )

        ev = result.evidence
        is_single_image = path_b is None
        evidence_payload = {
            "slot1_original": pil_to_base64(ev.get("original") or ev.get("before") or ev.get("optical")),
            "slot2_attention_or_diff": pil_to_base64(ev.get("attention") or ev.get("diff_heatmap") or ev.get("disagreement_map")),
            "slot3_reticle_or_sar": pil_to_base64(ev.get("box") or ev.get("diff_box") or ev.get("sar")),
            "slot4_after": pil_to_base64(ev.get("after") or ev.get("cir")),
            "is_single_image": is_single_image,
        }

        trace_data = result.trace.to_dict() if result.trace else None
        detected_objects = getattr(result, "detected_objects", [])

        return {
            "answer": result.answer,
            "evidence": evidence_payload,
            "detected_objects": detected_objects,
            "trace": trace_data,
            "trace_markdown": result.trace.to_markdown() if result.trace else "",
        }
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(path_a):
                os.remove(path_a)
            if path_b and os.path.exists(path_b):
                os.remove(path_b)
        except Exception:
            pass


@app.post("/api/detect")
async def detect_objects(
    target_classes: str = Form("all"),
    image: UploadFile = File(...),
):
    """Direct high-accuracy multi-object detection and precision bounding boxes."""
    content = await image.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    
    detected_objects = []
    boxed_img = None
    narrative = ""
    
    if getattr(agent_module, "_HAS_CLOUD_VLM", False) and agent_module.is_cloud_vlm_enabled():
        try:
            target_desc = "all objects (airplanes, storage tanks, ships, buildings, vehicles, runways, water bodies)" if target_classes == "all" else target_classes
            prompt = (
                f"You are an expert satellite remote sensing imagery analyst. "
                f"Detect and localize {target_desc} in this satellite image with high precision. "
                "For EACH detected object, output its category label and 2D bounding box [ymin, xmin, ymax, xmax] "
                "normalized from 0 to 1000 in JSON format:\n"
                "```json\n"
                "[\n"
                '  {"label": "airplane", "box_2d": [ymin, xmin, ymax, xmax], "confidence": 0.95}\n'
                "]\n"
                "```\n"
                "Include a comprehensive remote sensing detection summary."
            )
            narrative = agent_module.call_cloud_vlm(prompt, pil_img)
            boxed_img, detected_objects = agent_module.parse_boxes_with_metadata(pil_img, narrative)
        except Exception as e:
            print(f"[detect] Cloud detection fallback: {e}")

    if not detected_objects:
        try:
            from satquery.models.engine import RemoteSensingVLMEngine
            cv_narrative, cv_boxes = RemoteSensingVLMEngine.get_instance().detect_all_objects(pil_img, target_classes)
            detected_objects = cv_boxes[:20]
            if not narrative:
                narrative = cv_narrative
            boxed_img = agent_module.draw_bounding_boxes(pil_img, detected_objects)
        except Exception as e:
            print(f"[detect] CV fallback error: {e}")

    if boxed_img is None:
        boxed_img = pil_img

    return {
        "summary": narrative,
        "total_detected": len(detected_objects),
        "detected_objects": detected_objects,
        "annotated_image": pil_to_base64(boxed_img),
        "original_image": pil_to_base64(pil_img),
    }


@app.post("/api/spectral-indices")
async def calculate_spectral_indices(
    index_type: str = Form("ndvi"),
    image: UploadFile = File(...),
):
    """Compute scientific remote sensing spectral indices (NDVI, NDWI, NDBI, False-Color CIR)."""
    from satquery.core.spectral_indices import compute_spectral_index
    
    content = await image.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    
    result_img, stats = compute_spectral_index(pil_img, index_type=index_type)
    
    return {
        "index_type": index_type,
        "processed_image": pil_to_base64(result_img),
        "statistics": stats,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
