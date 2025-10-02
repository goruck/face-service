"""Face Service: FastAPI + InsightFace + ONNX Runtime."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict, cast

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[reportMissingTypeStubs]
import uvicorn
from fastapi import FastAPI, File, UploadFile
from insightface.app import FaceAnalysis  # type: ignore[reportMissingTypeStubs]

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Face Service")

# ------------------------------------------------------------------------------
# Config: GPU vs CPU
# ------------------------------------------------------------------------------
GPU_ID: int = int(os.getenv("GPU_ID", "0"))
USE_CPU: bool = os.getenv("USE_CPU", "false").lower() == "true"
MODEL_NAME: str = os.getenv("FACE_MODEL", "buffalo_l")  # allow swapping bundles

if USE_CPU:
    providers: list[str] = ["CPUExecutionProvider"]
    ctx_id: int = -1
    backend: str = "CPU"
else:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = GPU_ID
    backend = f"GPU:{GPU_ID}"


# ------------------------------------------------------------------------------
# InsightFace initialization
# ------------------------------------------------------------------------------
class _HasPrepare(Protocol):
    def prepare(
        self,
        ctx_id: int,
        det_thresh: float = ...,
        det_size: tuple[int, int] = ...,
    ) -> None: ...


face_app = FaceAnalysis(name=MODEL_NAME, providers=providers)  # runtime object
face_app_typed = cast(_HasPrepare, face_app)  # for static typing only

det_size: tuple[int, int] = (640, 640)
face_app_typed.prepare(ctx_id=ctx_id, det_size=det_size)


# ------------------------------------------------------------------------------
# Collect ORT provider info per sub-model (typed)
# ------------------------------------------------------------------------------
class ProviderInfo(TypedDict, total=False):
    providers: list[str]
    active_provider: str | None


model_providers: dict[str, ProviderInfo] = {}

# Treat untyped third-party attribute as a typed mapping: str -> Any
models_map: Mapping[str, Any] = cast(Mapping[str, Any], getattr(face_app, "models", {}))

for model_name, model in models_map.items():
    # Some sub-models expose .session (onnxruntime.InferenceSession)
    sess: Any = getattr(model, "session", None)
    if isinstance(sess, ort.InferenceSession):
        provs = list(sess.get_providers())
        model_providers[model_name] = {
            "providers": provs,
            "active_provider": provs[0] if provs else None,
        }
    else:
        model_providers[model_name] = {"providers": [], "active_provider": None}


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
upload_file_param = File(...)


@app.post("/analyze")
async def analyze(file: UploadFile = upload_file_param) -> dict[str, list[dict[str, Any]]]:
    """Analyze an uploaded image; return faces with bbox/score/embedding."""
    file_bytes: bytes = await file.read()

    # --- Decode with OpenCV ---
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        logging.warning("Failed to decode image")
        return {"faces": []}

    # --- Run InsightFace detection ---
    faces = face_app.get(img)  # type: ignore[attr-defined]
    logging.info("Detected %d faces", len(faces))

    results: list[dict[str, Any]] = []
    for i, face in enumerate(faces):
        # Some attributes are numpy arrays; coerce to native types for JSON
        bbox = getattr(face, "bbox", None)
        det_score = getattr(face, "det_score", None)
        embedding = getattr(face, "embedding", None)

        logging.info("Face %d: bbox=%s score=%s", i, bbox, det_score)

        results.append(
            {
                "bbox": bbox.tolist() if bbox is not None else None,
                "score": float(det_score) if det_score is not None else None,
                "embedding": embedding.tolist() if embedding is not None else None,
            }
        )

    return {"faces": results}


@app.get("/status")
async def status() -> dict[str, Any]:
    """Return runtime status, providers, and model info."""
    return {
        "requested_backend": "CPU" if USE_CPU else "GPU",
        "backend_detail": backend,
        "gpu_id": None if USE_CPU else GPU_ID,
        "face_model": MODEL_NAME,
        "available_providers": list(ort.get_available_providers()),
        "submodel_providers": model_providers,
    }


# ------------------------------------------------------------------------------
# Entrypoint (manual run)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
