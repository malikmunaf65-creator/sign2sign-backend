"""
Sign2Sign AI — Railway Backend (v3)
- Firebase init is lazy (won't crash startup if credentials not set yet)
- Landmarks come from browser MediaPipe JS (no opencv/mediapipe on server)
"""

import os, time, logging, json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sign2sign")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "")
FIREBASE_CREDS  = os.getenv("FIREBASE_CREDENTIALS_JSON", "")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Sign2Sign AI", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Firebase — lazy init (does NOT run at startup, only on first request) ─────
_firebase_ready = False

def get_firebase_auth():
    global _firebase_ready
    if _firebase_ready:
        from firebase_admin import auth as fb_auth
        return fb_auth
    try:
        import firebase_admin
        from firebase_admin import credentials, auth as fb_auth
        if not firebase_admin._apps:
            creds_str = FIREBASE_CREDS.strip()
            if not creds_str or creds_str in ("{}", ""):
                raise ValueError(
                    "FIREBASE_CREDENTIALS_JSON is empty. "
                    "Paste your serviceAccountKey.json content into Railway Variables."
                )
            cred_dict = json.loads(creds_str)
            if cred_dict.get("type") != "service_account":
                raise ValueError(
                    "FIREBASE_CREDENTIALS_JSON must be a service_account JSON. "
                    "Download it from Firebase Console → Project Settings → Service Accounts."
                )
            firebase_admin.initialize_app(credentials.Certificate(cred_dict))
        _firebase_ready = True
        log.info("Firebase initialized successfully")
        return fb_auth
    except Exception as e:
        log.error(f"Firebase init failed: {e}")
        raise HTTPException(status_code=503, detail=f"Auth service not configured: {e}")

# ── Supabase — optional ───────────────────────────────────────────────────────
_supabase = None
def get_supabase():
    global _supabase
    if _supabase is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            log.warning(f"Supabase init failed: {e}")
    return _supabase

# ── Load model at startup ─────────────────────────────────────────────────────
log.info("Loading TF model…")
MODEL  = tf.keras.models.load_model("sign2sign_v5_final.keras")
LABELS = np.load("label_classes.npy", allow_pickle=True)
log.info(f"Model ready — {len(LABELS)} classes: {list(LABELS)}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class LandmarkPayload(BaseModel):
    landmarks: list[float]   # 63 floats from MediaPipe JS in browser

class PredictionResponse(BaseModel):
    success: bool
    top_sign: str
    confidence: float
    top_five: list[dict]
    latency_ms: float

# ── Auth dependency ───────────────────────────────────────────────────────────
async def verify_token(request: Request) -> dict:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")
    fb_auth = get_firebase_auth()
    try:
        return fb_auth.verify_id_token(header.split("Bearer ")[1])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Sign2Sign AI v3 ✅", "classes": int(len(LABELS))}

@app.get("/health")
def health():
    # NOTE: Does NOT require Firebase — lets Railway healthcheck pass
    # even before Firebase credentials are configured
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: LandmarkPayload, token: dict = Depends(verify_token)):
    t0 = time.perf_counter()

    if len(payload.landmarks) != 63:
        raise HTTPException(status_code=400,
            detail=f"Expected 63 landmark floats, got {len(payload.landmarks)}")

    vec   = np.array(payload.landmarks, dtype=np.float32).reshape(1, -1)
    probs = MODEL.predict(vec, verbose=0)[0]

    top_idx  = np.argsort(probs)[::-1][:5]
    top_five = [{"sign": str(LABELS[i]), "confidence": round(float(probs[i]), 4)}
                for i in top_idx]

    # Log to Supabase (best-effort, won't crash if not configured)
    sb = get_supabase()
    if sb:
        try:
            sb.table("predictions").insert({
                "user_id":    token["uid"],
                "top_sign":   top_five[0]["sign"],
                "confidence": top_five[0]["confidence"],
                "all_preds":  top_five,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as e:
            log.warning(f"Supabase log failed: {e}")

    return PredictionResponse(
        success=True,
        top_sign=top_five[0]["sign"],
        confidence=top_five[0]["confidence"],
        top_five=top_five,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

@app.get("/history")
async def get_history(limit: int = 50, token: dict = Depends(verify_token)):
    sb = get_supabase()
    if not sb:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    result = (sb.table("predictions").select("*")
              .eq("user_id", token["uid"]).order("created_at", desc=True)
              .limit(limit).execute())
    return {"predictions": result.data}