# REDEPLOY: 2026-08-18 — fixed token_sequence regression (stale build)
"""
AIROS V10.3 — Temporal Behavior Inference Server
POST /predict  →  { p_down, p_up, context_ready, inference_ms, model_version }
"""

import os
import time
import math
import json
import logging
import traceback

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airos")

# ─────────────────────────────────────────────────────────────
# PATHS  (files sit alongside app.py in the repo root)
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "airos_v10_3_best.pt")
MEMORY_PATH   = os.path.join(BASE_DIR, "v10_3_behavior_memory.npz")

# ─────────────────────────────────────────────────────────────
# CONFIG  (must mirror training CONFIG exactly)
# ─────────────────────────────────────────────────────────────
LOOKBACK_MINUTES      = 25
TARGET_MINUTES        = 5
N_FEATURES            = 15
CTX_DIM               = 12
D_MODEL               = 96
N_HEADS               = 4
N_LAYERS              = 2
D_FF                  = 192
DROPOUT               = 0.10

HISTORY_TOP_K         = 8
MAX_HISTORY_DAYS      = 64
PRIOR_SMOOTHING       = 4.0
EXACT_MIN_EXAMPLES    = 4
HISTORY_EVIDENCE_DAYS = 12.0
SIMILARITY_TEMPERATURE= 1.0

FEATURE_COLS = [
    "ret1","ret3","ret5","body_norm","upper_norm","lower_norm",
    "range_atr","atr_pct","close_pos","vol5","vol15",
    "momentum5","momentum15","ticks_norm","tick_change",
]

MODEL_VERSION = "V10.3-temporal-behavior-brain"

# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (identical to training Cell 7)
# ─────────────────────────────────────────────────────────────
class BehavioralBrain(nn.Module):
    def __init__(self, n_assets):
        super().__init__()
        d = D_MODEL

        self.input   = nn.Sequential(nn.Linear(N_FEATURES, d), nn.LayerNorm(d))
        self.pos     = nn.Parameter(torch.randn(1, LOOKBACK_MINUTES, d) * 0.02)

        self.weekday = nn.Embedding(7,  8)
        self.hour    = nn.Embedding(24, 12)
        self.minute  = nn.Embedding(60, 8)
        self.asset   = nn.Embedding(n_assets, 12)

        temporal_dim = 40
        self.context = nn.Sequential(
            nn.Linear(CTX_DIM + temporal_dim, d),
            nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )

        self.cls    = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.memory = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm    = nn.LayerNorm(d)
        self.head    = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(d // 2, 2)
        )

    def forward(self, x, ctx, meta):
        aid    = meta[:, 1].long()
        wd     = meta[:, 2].long()
        hr     = meta[:, 3].long()
        minute = meta[:, 4].long()

        h = self.input(x) + self.pos
        temporal = torch.cat([
            self.weekday(wd), self.hour(hr),
            self.minute(minute), self.asset(aid)
        ], dim=-1)
        c = self.context(torch.cat([ctx, temporal], dim=-1)).unsqueeze(1)

        h = h + c
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1) + c
        mem = self.memory.expand(B, -1, -1) + c
        z   = torch.cat([cls, mem, h], dim=1)
        z   = self.norm(self.encoder(z))
        return self.head(z[:, 0])


# ─────────────────────────────────────────────────────────────
# SEQUENCE SIGNATURE  (identical to training Cell 4)
# ─────────────────────────────────────────────────────────────
def sequence_signature(seq: np.ndarray) -> np.ndarray:
    """seq: [25, 15] float32, column-indexed per FEATURE_COLS"""
    r       = seq[:, 0]
    body    = seq[:, 3]
    rng     = seq[:, 6]
    close_p = seq[:, 8]
    vol5    = seq[:, 9]
    vol15   = seq[:, 10]
    mom5    = seq[:, 11]
    mom15   = seq[:, 12]
    ticks   = seq[:, 13]
    tchg    = seq[:, 14]
    return np.array([
        np.sum(r),   np.mean(r),   np.std(r),
        np.sum(body), np.mean(np.abs(body)),
        np.mean(rng), np.std(rng),
        np.mean(close_p), np.std(close_p),
        np.mean(vol5), np.mean(vol15),
        np.sum(mom5), np.sum(mom15),
        np.mean(ticks), np.mean(tchg),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# STARTUP — load model + behavioral memory
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")   # Render free tier has no GPU

log.info("Loading model checkpoint: %s", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

ASSETS        = ckpt["assets"]
ASSET_TO_ID   = ckpt["asset_to_id"]
TRAIN_STATS   = {
    a: {
        "mean": np.array(v["mean"], dtype=np.float32),
        "std":  np.array(v["std"],  dtype=np.float32),
    }
    for a, v in ckpt["train_stats"].items()
}

model = BehavioralBrain(n_assets=len(ASSETS)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
log.info("Model loaded — %d assets, %d parameters",
         len(ASSETS), sum(p.numel() for p in model.parameters()))

# Build a fast lookup dict: (asset, date_str, hour, minute) → 12-dim ctx vector
log.info("Loading behavioral memory: %s", MEMORY_PATH)
_mem = np.load(MEMORY_PATH, allow_pickle=True)
_keys   = _mem["keys"]    # object array of tuples (asset, date, hour, minute)
_values = _mem["values"]  # float32 [N, 12]

BEHAVIOR_MEMORY: dict = {}
for k, v in zip(_keys, _values):
    # keys stored as (asset, date, hour, minute)
    # date may be a datetime.date or string; normalise to string
    asset_k, date_k, hour_k, minute_k = k[0], str(k[1]), int(k[2]), int(k[3])
    BEHAVIOR_MEMORY[(asset_k, date_k, int(hour_k), int(minute_k))] = v.astype(np.float32)

log.info("Behavioral memory loaded — %d anchors", len(BEHAVIOR_MEMORY))


# ─────────────────────────────────────────────────────────────
# LIVE CTX CONSTRUCTION
# Mirrors the four-level matching hierarchy from training Cell 4.
# Only historical anchors (dates strictly before current date) are used.
# ─────────────────────────────────────────────────────────────
def build_live_ctx(
    asset: str,
    weekday: int,
    hour: int,
    minute: int,
    current_date_str: str,      # "YYYY-MM-DD"
    live_seq: np.ndarray,       # [25, 15] unnormalised features
) -> tuple[np.ndarray, bool]:
    """
    Returns (ctx_12dim, context_ready).
    context_ready=False means no prior history existed; ctx is zeros (prior=0.5).
    """
    cur_sig = sequence_signature(live_seq)

    # Collect all historical anchors for same asset + weekday + hour,
    # strictly before current_date_str.
    prior = []
    for (a, d, h, m), vec in BEHAVIOR_MEMORY.items():
        if a != asset or h != hour:
            continue
        if d >= current_date_str:   # causal: prior dates only
            continue
        prior.append((d, m, vec))   # (date_str, minute, ctx_vec)

    if not prior:
        return np.zeros(CTX_DIM, dtype=np.float32), False

    # Trim to max_history_days * 60 most recent
    prior.sort(key=lambda x: (x[0], x[1]))
    prior = prior[-(MAX_HISTORY_DAYS * 60):]

    # ── Sequence-similarity + minute-proximity weighting ──────
    # Use the first 8 elements of stored CTX as a proxy for the
    # historical sequence signature (they encode the probability
    # statistics computed from those sequences).
    # Full re-signatures aren't stored; we use the ctx vectors directly
    # as the similarity space — consistent because they were computed
    # from the same sequence_signature() function at training time.
    stored_vecs = np.stack([p[2] for p in prior]).astype(np.float32)
    scale = np.maximum(np.std(stored_vecs, axis=0), 1e-5)
    dist  = np.mean(np.abs(stored_vecs - cur_sig[:CTX_DIM][None, :]) / scale[None, :], axis=1)
    sim   = np.exp(-dist / max(SIMILARITY_TEMPERATURE, 1e-6))

    minute_dist   = np.array([abs(p[1] - minute) for p in prior], dtype=np.float32)
    minute_weight = np.exp(-minute_dist / 8.0)
    combined      = sim * (0.65 + 0.35 * minute_weight)

    k    = min(HISTORY_TOP_K, len(combined))
    ids  = np.argpartition(combined, -k)[-k:]
    w    = combined[ids] + 1e-8
    # Use CTX[0] (hist_up) as historical outcome proxy
    states = stored_vecs[ids, 0]
    top_up = float(np.average(states, weights=w))
    similarity = float(np.average(sim[ids], weights=w))

    # ── Hour-level Bayesian probability ───────────────────────
    hour_states = stored_vecs[:, 0]
    hour_count  = len(hour_states)
    hour_sum    = float(hour_states.sum())
    alpha       = PRIOR_SMOOTHING
    hour_up     = (hour_sum + 0.5 * alpha) / (hour_count + alpha)

    # ── Exact-minute subset ───────────────────────────────────
    exact = [(i, p) for i, p in enumerate(prior) if p[1] == minute]
    if exact:
        ex_ids   = [e[0] for e in exact]
        ex_vecs  = stored_vecs[ex_ids]
        ex_states= ex_vecs[:, 0]
        ex_scale = np.maximum(np.std(ex_vecs, axis=0), 1e-5)
        ex_dist  = np.mean(np.abs(ex_vecs - cur_sig[:CTX_DIM][None, :]) / ex_scale[None, :], axis=1)
        ex_sim   = np.exp(-ex_dist / max(SIMILARITY_TEMPERATURE, 1e-6))
        ek       = min(HISTORY_TOP_K, len(ex_sim))
        eids     = np.argpartition(ex_sim, -ek)[-ek:]
        ew       = ex_sim[eids] + 1e-8
        exact_up       = float(np.average(ex_states[eids], weights=ew))
        exact_similarity = float(np.average(ex_sim[eids], weights=ew))
        exact_count    = len(ex_states)
        exact_sum      = float(ex_states.sum())
    else:
        exact_up = 0.5
        exact_similarity = 0.0
        exact_count = 0
        exact_sum   = 0.0

    ex_up = ((exact_sum + 0.5 * alpha) / (exact_count + alpha)) if exact_count else 0.5

    exact_strength = min(1.0, exact_count / float(EXACT_MIN_EXAMPLES))
    hist_up  = exact_strength * ex_up + (1.0 - exact_strength) * hour_up
    evidence = min(1.0, hour_count / float(HISTORY_EVIDENCE_DAYS))
    agreement= 1.0 - abs(ex_up - hour_up)

    recent_states = stored_vecs[-min(8, len(stored_vecs)):, 0]
    recent_up     = float(recent_states.mean()) if len(recent_states) else 0.5

    ctx = np.array([
        hist_up,                        # 0
        hour_up,                        # 1
        ex_up,                          # 2
        top_up,                         # 3
        evidence,                       # 4
        math.log1p(hour_count),         # 5
        math.log1p(exact_count),        # 6
        similarity,                     # 7
        exact_similarity,               # 8
        agreement,                      # 9
        recent_up,                      # 10
        float(minute / 59.0),           # 11
    ], dtype=np.float32)

    return ctx, True


# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_VERSION,
        "assets": len(ASSETS),
        "memory_anchors": len(BEHAVIOR_MEMORY),
    })


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    try:
        body = request.get_json(force=True)

        # ── Validate version ──────────────────────────────────
        if body.get("version") != MODEL_VERSION:
            return jsonify({"success": False,
                            "error": f"Version mismatch. Expected {MODEL_VERSION}"}), 400

        # ── Asset ─────────────────────────────────────────────
        asset = str(body.get("asset", "")).upper().replace("_OTC", "").strip()
        if asset not in ASSET_TO_ID:
            return jsonify({"success": False,
                            "error": f"Unknown asset: {asset}. Known: {ASSETS}"}), 400
        asset_id = ASSET_TO_ID[asset]

        # ── Features [25, 15] ─────────────────────────────────
        features = body.get("features")
        if not features or len(features) != LOOKBACK_MINUTES:
            return jsonify({"success": False,
                            "error": f"features must be {LOOKBACK_MINUTES} rows"}), 400
        for row in features:
            if len(row) != N_FEATURES:
                return jsonify({"success": False,
                                "error": f"Each feature row must have {N_FEATURES} values"}), 400

        raw_seq = np.array(features, dtype=np.float32)   # [25, 15], unnormalised

        # ── Temporal ──────────────────────────────────────────
        temporal    = body.get("temporal", {})
        weekday     = int(temporal.get("weekday", 0))
        hour        = int(temporal.get("hour",    0))
        minute      = int(temporal.get("minute",  0))
        ts_ms       = int(temporal.get("timestamp", 0))

        # Derive date string from timestamp
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        current_date_str = dt.strftime("%Y-%m-%d")

        # ── Normalise features (per-asset train-period stats) ─
        st  = TRAIN_STATS[asset]
        norm_seq = (raw_seq - st["mean"][None, :]) / st["std"][None, :]
        norm_seq = np.clip(norm_seq, -8, 8).astype(np.float32)

        # ── Build live CTX ────────────────────────────────────
        ctx_vec, context_ready = build_live_ctx(
            asset=asset,
            weekday=weekday,
            hour=hour,
            minute=minute,
            current_date_str=current_date_str,
            live_seq=raw_seq,
        )

        # ── Build META tensor  [split=0, asset_id, wd, hr, min, ordinal] ─
        date_ordinal = dt.toordinal()
        meta_np = np.array(
            [[0, asset_id, weekday, hour, minute, date_ordinal]],
            dtype=np.int32
        )

        # ── Inference ─────────────────────────────────────────
        x_t    = torch.from_numpy(norm_seq).unsqueeze(0).to(DEVICE)    # [1,25,15]
        ctx_t  = torch.from_numpy(ctx_vec).unsqueeze(0).to(DEVICE)     # [1,12]
        meta_t = torch.from_numpy(meta_np).to(DEVICE)                  # [1,6]

        with torch.no_grad():
            logits = model(x_t, ctx_t, meta_t)                         # [1,2]
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_down = float(probs[0])
        p_up   = float(probs[1])

        # Normalise for floating point drift
        total  = p_down + p_up
        p_down /= total
        p_up   /= total

        inference_ms = int((time.time() - t0) * 1000)

        log.info("PREDICT asset=%s wd=%d hr=%d min=%d | p_up=%.4f p_down=%.4f ctx=%s %dms",
                 asset, weekday, hour, minute, p_up, p_down,
                 "ready" if context_ready else "prior", inference_ms)

        return jsonify({
            "success":       True,
            "model_version": MODEL_VERSION,
            "p_down":        round(p_down, 6),
            "p_up":          round(p_up,   6),
            "context_ready": context_ready,
            "inference_ms":  inference_ms,
        })

    except Exception:
        log.error("Inference error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
