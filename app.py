import logging
import os

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, UploadFile
from insightface.app import FaceAnalysis

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------------------------------------
# Config: GPU vs CPU
# -------------------------------------------------------------------
GPU_ID = int(os.getenv("GPU_ID", "0"))
USE_CPU = os.getenv("USE_CPU", "false").lower() == "true"
MODEL_NAME = os.getenv("FACE_MODEL", "buffalo_l")  # allow swapping model bundles

if USE_CPU:
    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    backend = "CPU"
else:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = GPU_ID
    backend = f"GPU:{GPU_ID}"

# -------------------------------------------------------------------
# Initialize InsightFace
# -------------------------------------------------------------------
face_app = FaceAnalysis(name=MODEL_NAME, providers=providers)
face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

# Collect ORT provider info per sub-model
model_providers = {}
for name, model in face_app.models.items():
    sess = getattr(model, "session", None)
    if isinstance(sess, ort.InferenceSession):
        model_providers[name] = {
            "providers": sess.get_providers(),
            "active_provider": sess.get_providers()[0] if sess.get_providers() else None,
        }
    else:
        model_providers[name] = {"providers": [], "active_provider": None}

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Read uploaded bytes
    file_bytes = await file.read()

    # --- Decode with OpenCV ---
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        logging.warning("Failed to decode image")
        return {"faces": []}

    # --- Run InsightFace detection ---
    faces = face_app.get(img)
    logging.info("Detected %d faces", len(faces))
    for i, f in enumerate(faces):
        logging.info("Face %d: bbox=%s score=%.3f", i, f.bbox, f.det_score)

    results = []
    for i, face in enumerate(faces):
        results.append({
            "bbox": face.bbox.tolist(),
            "score": float(face.det_score),
            "embedding": face.embedding.tolist()
        })

    return {"faces": results}


@app.get("/status")
async def status():
    return {
        "requested_backend": "GPU" if not USE_CPU else "CPU",
        "gpu_id": GPU_ID if not USE_CPU else None,
        "face_model": MODEL_NAME,
        "available_providers": ort.get_available_providers(),
        "submodel_providers": model_providers,
    }

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
