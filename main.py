"""
Sign2Sign AI — Railway Backend (v4)
Runs in two modes:
  DEV  mode — no Firebase/Supabase needed, auth is skipped
  PROD mode — real Firebase credentials required
Set FIREBASE_CREDENTIALS_JSON in Railway Variables to switch to PROD.
"""

import os, time, logging, json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sign2sign")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
FIREBASE_CREDS  = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
SUPABASE_URL    = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

# Detect whether real credentials have been provided
def _is_real_firebase():
    if not FIREBASE_CREDS or FIREBASE_CREDS in ("{}", "dummy", ""):
        return False
    try:
        d = json.loads(FIREBASE_CREDS)
        return d.get("type") == "service_account"
    except Exception:
        return False

def _is_real_supabase():
    return bool(SUPABASE_URL and SUPABASE_KEY
                and "dummy" not in SUPABASE_URL
                and "dummy" not in SUPABASE_KEY)

AUTH_ENABLED     = _is_real_firebase()
SUPABASE_ENABLED = _is_real_supabase()

log.info(f"Auth enabled:     {AUTH_ENABLED}")
log.info(f"Supabase enabled: {SUPABASE_ENABLED}")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Sign2Sign AI", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Firebase (only if real creds exist) ───────────────────────────────────────
if AUTH_ENABLED:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(json.loads(FIREBASE_CREDS)))
    log.info("Firebase Admin initialized ✅")

# ── Supabase (only if real creds exist) ───────────────────────────────────────
supabase = None
if SUPABASE_ENABLED:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase connected ✅")
    except Exception as e:
        log.warning(f"Supabase failed: {e}")

# ── Load model ────────────────────────────────────────────────────────────────
log.info("Loading TF model…")
MODEL  = tf.keras.models.load_model("sign2sign_v5_final.keras")
LABELS = np.load("label_classes.npy", allow_pickle=True)
log.info(f"Model ready — {len(LABELS)} classes ✅")

# ── Schemas ───────────────────────────────────────────────────────────────────
class LandmarkPayload(BaseModel):
    landmarks: list[float]   # 63 floats from MediaPipe JS in browser

class PredictionResponse(BaseModel):
    success: bool
    top_sign: str
    confidence: float
    top_five: list[dict]
    latency_ms: float
    auth_mode: str           # "dev" or "prod" — useful for debugging

# ── Token verification (skipped in dev mode) ──────────────────────────────────
def get_uid(request: Request) -> str:
    """Returns Firebase UID in prod, 'dev-user' in dev mode."""
    if not AUTH_ENABLED:
        return "dev-user"
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return "dev-user"
    try:
        decoded = firebase_auth.verify_id_token(header.split("Bearer ")[1])
        return decoded["uid"]
    except Exception:
        return "dev-user"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "Sign2Sign AI ✅",
        "auth_mode": "prod" if AUTH_ENABLED else "dev",
        "supabase": SUPABASE_ENABLED,
        "classes": int(len(LABELS)),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: LandmarkPayload, request: Request):
    t0  = time.perf_counter()
    uid = get_uid(request)

    if len(payload.landmarks) != 63:
        from fastapi import HTTPException
        raise HTTPException(status_code=400,
            detail=f"Expected 63 landmark values, got {len(payload.landmarks)}")

    vec   = np.array(payload.landmarks, dtype=np.float32).reshape(1, -1)
    probs = MODEL.predict(vec, verbose=0)[0]

    top_idx  = np.argsort(probs)[::-1][:5]
    top_five = [{"sign": str(LABELS[i]), "confidence": round(float(probs[i]), 4)}
                for i in top_idx]

    # Log to Supabase if available
    if supabase:
        try:
            supabase.table("predictions").insert({
                "user_id":    uid,
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
        auth_mode="prod" if AUTH_ENABLED else "dev",
    )

@app.get("/history")
async def get_history(limit: int = 50, request: Request = None):
    if not supabase:
        return {"predictions": [], "note": "Supabase not configured yet"}
    uid = get_uid(request)
    result = (supabase.table("predictions").select("*")
              .eq("user_id", uid).order("created_at", desc=True)
              .limit(limit).execute())
    return {"predictions": result.data}
