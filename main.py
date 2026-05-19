"""
Sign2Sign AI — Railway Backend (v2)
Landmarks extracted in browser via MediaPipe JS.
This server only receives 63 floats and runs TF inference.
No OpenCV. No MediaPipe. No libGL. No system deps.
"""

import os
import time
import logging
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

from supabase import create_client, Client as SupabaseClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sign2sign")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "")
FIREBASE_CREDS  = os.getenv("FIREBASE_CREDENTIALS_JSON", "")

app = FastAPI(title="Sign2Sign AI Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_firebase():
    if not firebase_admin._apps:
        if FIREBASE_CREDS:
            import json
            cred = credentials.Certificate(json.loads(FIREBASE_CREDS))
        else:
            cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    log.info("Firebase Admin initialized")

init_firebase()

supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

log.info("Loading model...")
MODEL  = tf.keras.models.load_model("sign2sign_v5_final.keras")
LABELS = np.load("label_classes.npy", allow_pickle=True)
log.info(f"Model loaded. {len(LABELS)} classes.")

class LandmarkPayload(BaseModel):
    landmarks: list[float]

class PredictionResponse(BaseModel):
    success: bool
    top_sign: str
    confidence: float
    top_five: list[dict]
    latency_ms: float

async def verify_token(request: Request) -> dict:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        return firebase_auth.verify_id_token(header.split("Bearer ")[1])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

def log_prediction(user_id, top_sign, confidence, all_preds):
    if not supabase:
        return
    try:
        supabase.table("predictions").insert({
            "user_id": user_id, "top_sign": top_sign,
            "confidence": confidence, "all_preds": all_preds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"Supabase log failed: {e}")

@app.get("/")
def root():
    return {"status": "Sign2Sign AI v2 running", "model": "sign2sign_v5_final"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: LandmarkPayload, token: dict = Depends(verify_token)):
    t0 = time.perf_counter()
    if len(payload.landmarks) != 63:
        raise HTTPException(status_code=400, detail=f"Expected 63 values, got {len(payload.landmarks)}")
    vec   = np.array(payload.landmarks, dtype=np.float32).reshape(1, -1)
    probs = MODEL.predict(vec, verbose=0)[0]
    top_idx  = np.argsort(probs)[::-1][:5]
    top_five = [{"sign": str(LABELS[i]), "confidence": round(float(probs[i]), 4)} for i in top_idx]
    log_prediction(token["uid"], top_five[0]["sign"], top_five[0]["confidence"], top_five)
    return PredictionResponse(
        success=True, top_sign=top_five[0]["sign"],
        confidence=top_five[0]["confidence"], top_five=top_five,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

@app.get("/history")
async def get_history(limit: int = 50, token: dict = Depends(verify_token)):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    result = (supabase.table("predictions").select("*")
              .eq("user_id", token["uid"]).order("created_at", desc=True).limit(limit).execute())
    return {"predictions": result.data}
