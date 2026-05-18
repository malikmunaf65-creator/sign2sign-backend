"""
Sign2Sign AI — Railway Backend
FastAPI server: receives webcam frames, runs MediaPipe + TensorFlow,
verifies Firebase JWT, logs predictions to Supabase.
"""

import os
import io
import time
import base64
import logging
from datetime import datetime, timezone

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

from supabase import create_client, Client as SupabaseClient

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sign2sign")

# ── Environment variables (set these in Railway dashboard) ────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "")  # service role key
FIREBASE_CREDS  = os.getenv("FIREBASE_CREDENTIALS_JSON", "")  # full JSON string

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Sign2Sign AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # e.g. ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Firebase Admin SDK init ───────────────────────────────────────────────────
def init_firebase():
    if FIREBASE_CREDS:
        import json
        cred_dict = json.loads(FIREBASE_CREDS)
        cred = credentials.Certificate(cred_dict)
    else:
        # Local dev: place serviceAccountKey.json in project root
        cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    log.info("Firebase Admin initialized")

init_firebase()

# ── Supabase client ───────────────────────────────────────────────────────────
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# ── Load TensorFlow model ─────────────────────────────────────────────────────
log.info("Loading TensorFlow model…")
MODEL = tf.keras.models.load_model("sign2sign_v5_final.keras")
LABELS = np.load("label_classes.npy", allow_pickle=True)
log.info(f"Model loaded. Classes: {list(LABELS)}")

# ── MediaPipe Hands ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
HANDS = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class FramePayload(BaseModel):
    # Base64-encoded JPEG frame captured from the webcam
    frame: str          # "data:image/jpeg;base64,/9j/4AAQ..."  or raw base64

class PredictionResponse(BaseModel):
    success: bool
    top_sign: str
    confidence: float
    top_five: list[dict]   # [{"sign": "A", "confidence": 0.92}, ...]
    hand_detected: bool
    latency_ms: float

# ── Firebase token dependency ─────────────────────────────────────────────────
async def verify_firebase_token(request: Request) -> dict:
    """
    Reads the Authorization: Bearer <token> header,
    verifies it with Firebase Admin SDK,
    returns the decoded token payload (contains uid, email, etc.)
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    id_token = auth_header.split("Bearer ")[1]
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Firebase token: {str(e)}")

# ── Landmark extraction ───────────────────────────────────────────────────────
def decode_frame(b64_frame: str) -> np.ndarray:
    """Convert base64 string → OpenCV BGR image array."""
    # Strip data URI prefix if present
    if "," in b64_frame:
        b64_frame = b64_frame.split(",")[1]
    img_bytes = base64.b64decode(b64_frame)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)   # RGB numpy array

def extract_landmarks(rgb_img: np.ndarray):
    """Run MediaPipe on RGB image → (63,) float32 array or None."""
    results = HANDS.process(rgb_img)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    coords = []
    for point in lm.landmark:
        coords.extend([point.x, point.y, point.z])   # 21 × 3 = 63 features
    return np.array(coords, dtype=np.float32)

# ── Log to Supabase ───────────────────────────────────────────────────────────
def log_prediction(user_id: str, top_sign: str, confidence: float, all_preds: list):
    """Insert one row into the predictions table (fire-and-forget)."""
    if not supabase:
        return
    try:
        supabase.table("predictions").insert({
            "user_id":    user_id,
            "top_sign":   top_sign,
            "confidence": confidence,
            "all_preds":  all_preds,          # stored as JSONB
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"Supabase log failed: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Sign2Sign AI backend is running ✅"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: FramePayload,
    token: dict = Depends(verify_firebase_token),
):
    """
    Main prediction endpoint.
    Expects a base64 JPEG frame.
    Returns top sign + confidence + top-5 breakdown.
    """
    t_start = time.perf_counter()

    # 1. Decode image
    try:
        rgb_img = decode_frame(payload.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {e}")

    # 2. Extract landmarks
    landmarks = extract_landmarks(rgb_img)
    if landmarks is None:
        return PredictionResponse(
            success=True,
            top_sign="none",
            confidence=0.0,
            top_five=[],
            hand_detected=False,
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
        )

    # 3. Model inference
    vec   = landmarks.reshape(1, -1)
    probs = MODEL.predict(vec, verbose=0)[0]   # (28,)

    # 4. Build top-5
    top_idx  = np.argsort(probs)[::-1][:5]
    top_five = [
        {"sign": str(LABELS[i]), "confidence": round(float(probs[i]), 4)}
        for i in top_idx
    ]
    top_sign   = top_five[0]["sign"]
    top_conf   = top_five[0]["confidence"]

    # 5. Log to Supabase (non-blocking, best-effort)
    log_prediction(
        user_id=token["uid"],
        top_sign=top_sign,
        confidence=top_conf,
        all_preds=top_five,
    )

    latency = round((time.perf_counter() - t_start) * 1000, 2)
    log.info(f"uid={token['uid']} → {top_sign} ({top_conf:.2%}) in {latency}ms")

    return PredictionResponse(
        success=True,
        top_sign=top_sign,
        confidence=top_conf,
        top_five=top_five,
        hand_detected=True,
        latency_ms=latency,
    )


@app.get("/history")
async def get_history(
    limit: int = 50,
    token: dict = Depends(verify_firebase_token),
):
    """Return last N predictions for the authenticated user."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")

    result = (
        supabase.table("predictions")
        .select("*")
        .eq("user_id", token["uid"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"predictions": result.data}
