# REDEPLOY: 2026-08-18 — indexed behavioral-memory retrieval
"""
AIROS V10.3 — Temporal Behavior Inference Server
POST /predict  →  { p_down, p_up, context_ready, inference_ms, model_version }

Behavioral-memory retrieval is indexed at startup. Live prediction never scans
all historical anchors; it directly selects the current asset + weekday + hour
bucket and then applies the causal date cutoff.
"""

import os
import time
import math
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airos")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "airos_v10_3_best.pt")
MEMORY_PATH   = os.path.join(BASE_DIR, "v10_3_behavior_memory.npz")

LOOKBACK_MINUTES       = 25
TARGET_MINUTES         = 5
N_FEATURES             = 15
CTX_DIM                = 12
D_MODEL                = 96
N_HEADS                = 4
N_LAYERS               = 2
D_FF                   = 192
DROPOUT                = 0.10

HISTORY_TOP_K          = 8
MAX_HISTORY_DAYS       = 64
PRIOR_SMOOTHING        = 4.0
EXACT_MIN_EXAMPLES     = 4
HISTORY_EVIDENCE_DAYS  = 12.0
SIMILARITY_TEMPERATURE = 1.0

FEATURE_COLS = [
    "ret1", "ret3", "ret5", "body_norm", "upper_norm", "lower_norm",
    "range_atr", "atr_pct", "close_pos", "vol5", "vol15",
    "momentum5", "momentum15", "ticks_norm", "tick_change",
]

MODEL_VERSION = "V10.3-temporal-behavior-brain"


class BehavioralBrain(nn.Module):
    def __init__(self, n_assets):
        super().__init__()
        d = D_MODEL

        self.input   = nn.Sequential(nn.Linear(N_FEATURES, d), nn.LayerNorm(d))
        self.pos     = nn.Parameter(torch.randn(1, LOOKBACK_MINUTES, d) * 0.02)

        self.weekday = nn.Embedding(7, 8)
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


def sequence_signature(seq: np.ndarray) -> np.ndarray:
    """seq: [25, 15] float32, column-indexed per FEATURE_COLS."""
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
        np.sum(r), np.mean(r), np.std(r),
        np.sum(body), np.mean(np.abs(body)),
        np.mean(rng), np.std(rng),
        np.mean(close_p), np.std(close_p),
        np.mean(vol5), np.mean(vol15),
        np.sum(mom5), np.sum(mom15),
        np.mean(ticks), np.mean(tchg),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# STARTUP — load model + build indexed behavioral memory
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")

log.info("Loading model checkpoint: %s", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

ASSETS      = ckpt["assets"]
ASSET_TO_ID = ckpt["asset_to_id"]
TRAIN_STATS = {
    a: {
        "mean": np.array(v["mean"], dtype=np.float32),
        "std":  np.array(v["std"], dtype=np.float32),
    }
    for a, v in ckpt["train_stats"].items()
}

model = BehavioralBrain(n_assets=len(ASSETS)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
log.info("Model loaded — %d assets, %d parameters",
         len(ASSETS), sum(p.numel() for p in model.parameters()))


# Index format:
#   (asset, weekday, hour) -> {
#       "dates":   sorted int32 ordinals,
#       "minutes": int8/int16 minutes,
#       "values":  float32 [N, 12]
#   }
#
# This preserves causal historical retrieval while avoiding the old
# O(543K) scan on every /predict request.
log.info("Loading and indexing behavioral memory: %s", MEMORY_PATH)
_mem = np.load(MEMORY_PATH, allow_pickle=True)
_keys   = _mem["keys"]
_values = _mem["values"]

_bucket_dates = {}
_bucket_minutes = {}
_bucket_values = {}

for k, v in zip(_keys, _values):
    asset_k = str(k[0]).upper().replace("_OTC", "").strip()
    date_s = str(k[1])
    hour_k = int(k[2])
    minute_k = int(k[3])

    # Training uses Python-style weekday semantics: Monday=0 ... Sunday=6.
    try:
        dt_k = datetime.strptime(date_s[:10], "%Y-%m-%d")
    except ValueError:
        continue

    wd_k = dt_k.weekday()
    bucket = (asset_k, wd_k, hour_k)

    _bucket_dates.setdefault(bucket, []).append(dt_k.toordinal())
    _bucket_minutes.setdefault(bucket, []).append(minute_k)
    _bucket_values.setdefault(bucket, []).append(np.asarray(v, dtype=np.float32))

# Convert every bucket once, sort chronologically, and discard the original
# 543K object arrays so they are not retained in memory.
BEHAVIOR_INDEX = {}
for bucket, dates in _bucket_dates.items():
    order = np.argsort(np.asarray(dates, dtype=np.int32), kind="stable")
    dates_arr = np.asarray(dates, dtype=np.int32)[order]
    minutes_arr = np.asarray(_bucket_minutes[bucket], dtype=np.int16)[order]
    values_arr = np.asarray(_bucket_values[bucket], dtype=np.float32)[order]
    BEHAVIOR_INDEX[bucket] = {
        "dates": dates_arr,
        "minutes": minutes_arr,
        "values": values_arr,
    }

# Release temporary full-memory structures after indexing.
del _mem, _keys, _values
_del_names = ("_bucket_dates", "_bucket_minutes", "_bucket_values")
for _name in _del_names:
    globals().pop(_name, None)

MEMORY_ANCHOR_COUNT = sum(len(v["dates"]) for v in BEHAVIOR_INDEX.values())
log.info("Behavioral memory indexed — %d anchors across %d asset/weekday/hour buckets",
         MEMORY_ANCHOR_COUNT, len(BEHAVIOR_INDEX))


def _get_causal_bucket(asset: str, weekday: int, hour: int, current_ordinal: int):
    """Return only prior-date anchors for the requested asset/weekday/hour."""
    bucket = BEHAVIOR_INDEX.get((asset, weekday, hour))
    if bucket is None:
        return None, None

    dates = bucket["dates"]
    # First position whose date is >= current date. Everything before it is causal.
    end = int(np.searchsorted(dates, current_ordinal, side="left"))
    if end <= 0:
        return None, None

    # At most MAX_HISTORY_DAYS*60 anchors, same as the previous implementation.
    start = max(0, end - (MAX_HISTORY_DAYS * 60))
    return bucket, (start, end)


# ─────────────────────────────────────────────────────────────
# LIVE CTX CONSTRUCTION
# Same context calculation, but retrieval is indexed by asset + weekday + hour.
# ─────────────────────────────────────────────────────────────
def build_live_ctx(
    asset: str,
    weekday: int,
    hour: int,
    minute: int,
    current_date_str: str,
    live_seq: np.ndarray,
) -> tuple[np.ndarray, bool]:
    cur_sig = sequence_signature(live_seq)

    try:
        current_ordinal = datetime.strptime(current_date_str, "%Y-%m-%d").toordinal()
    except ValueError:
        return np.zeros(CTX_DIM, dtype=np.float32), False

    bucket, bounds = _get_causal_bucket(asset, weekday, hour, current_ordinal)
    if bucket is None:
        return np.zeros(CTX_DIM, dtype=np.float32), False

    start, end = bounds
    stored_vecs = bucket["values"][start:end].astype(np.float32, copy=False)
    minutes = bucket["minutes"][start:end]

    if len(stored_vecs) == 0:
        return np.zeros(CTX_DIM, dtype=np.float32), False

    # Sequence-similarity + minute-proximity weighting.
    scale = np.maximum(np.std(stored_vecs, axis=0), 1e-5)
    dist = np.mean(
        np.abs(stored_vecs - cur_sig[:CTX_DIM][None, :]) / scale[None, :],
        axis=1,
    )
    sim = np.exp(-dist / max(SIMILARITY_TEMPERATURE, 1e-6))

    minute_dist = np.abs(minutes.astype(np.float32) - float(minute))
    minute_weight = np.exp(-minute_dist / 8.0)
    combined = sim * (0.65 + 0.35 * minute_weight)

    k = min(HISTORY_TOP_K, len(combined))
    ids = np.argpartition(combined, -k)[-k:]
    w = combined[ids] + 1e-8
    states = stored_vecs[ids, 0]
    top_up = float(np.average(states, weights=w))
    similarity = float(np.average(sim[ids], weights=w))

    # Hour-level Bayesian probability.
    hour_states = stored_vecs[:, 0]
    hour_count = len(hour_states)
    hour_sum = float(hour_states.sum())
    alpha = PRIOR_SMOOTHING
    hour_up = (hour_sum + 0.5 * alpha) / (hour_count + alpha)

    # Exact-minute subset.
    exact_mask = minutes == minute
    exact_indices = np.flatnonzero(exact_mask)

    if len(exact_indices):
        ex_vecs = stored_vecs[exact_indices]
        ex_states = ex_vecs[:, 0]
        ex_scale = np.maximum(np.std(ex_vecs, axis=0), 1e-5)
        ex_dist = np.mean(
            np.abs(ex_vecs - cur_sig[:CTX_DIM][None, :]) / ex_scale[None, :],
            axis=1,
        )
        ex_sim = np.exp(-ex_dist / max(SIMILARITY_TEMPERATURE, 1e-6))
        ek = min(HISTORY_TOP_K, len(ex_sim))
        eids = np.argpartition(ex_sim, -ek)[-ek:]
        ew = ex_sim[eids] + 1e-8
        exact_up = float(np.average(ex_states[eids], weights=ew))
        exact_similarity = float(np.average(ex_sim[eids], weights=ew))
        exact_count = len(ex_states)
        exact_sum = float(ex_states.sum())
    else:
        exact_up = 0.5
        exact_similarity = 0.0
        exact_count = 0
        exact_sum = 0.0

    ex_up = ((exact_sum + 0.5 * alpha) / (exact_count + alpha)) if exact_count else 0.5

    exact_strength = min(1.0, exact_count / float(EXACT_MIN_EXAMPLES))
    hist_up = exact_strength * ex_up + (1.0 - exact_strength) * hour_up
    evidence = min(1.0, hour_count / float(HISTORY_EVIDENCE_DAYS))
    agreement = 1.0 - abs(ex_up - hour_up)

    recent_states = stored_vecs[-min(8, len(stored_vecs)):, 0]
    recent_up = float(recent_states.mean()) if len(recent_states) else 0.5

    ctx = np.array([
        hist_up,
        hour_up,
        ex_up,
        top_up,
        evidence,
        math.log1p(hour_count),
        math.log1p(exact_count),
        similarity,
        exact_similarity,
        agreement,
        recent_up,
        float(minute / 59.0),
    ], dtype=np.float32)

    return ctx, True


app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_VERSION,
        "assets": len(ASSETS),
        "memory_anchors": MEMORY_ANCHOR_COUNT,
        "memory_buckets": len(BEHAVIOR_INDEX),
        "retrieval": "asset+weekday+hour indexed; causal date cutoff",
    })


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    try:
        body = request.get_json(force=True)

        if body.get("version") != MODEL_VERSION:
            return jsonify({
                "success": False,
                "error": f"Version mismatch. Expected {MODEL_VERSION}",
            }), 400

        asset = str(body.get("asset", "")).upper().replace("_OTC", "").strip()
        if asset not in ASSET_TO_ID:
            return jsonify({
                "success": False,
                "error": f"Unknown asset: {asset}. Known: {ASSETS}",
            }), 400
        asset_id = ASSET_TO_ID[asset]

        features = body.get("features")
        if not features or len(features) != LOOKBACK_MINUTES:
            return jsonify({
                "success": False,
                "error": f"features must be {LOOKBACK_MINUTES} rows",
            }), 400
        for row in features:
            if len(row) != N_FEATURES:
                return jsonify({
                    "success": False,
                    "error": f"Each feature row must have {N_FEATURES} values",
                }), 400

        raw_seq = np.array(features, dtype=np.float32)

        temporal = body.get("temporal", {})
        weekday = int(temporal.get("weekday", 0))
        hour = int(temporal.get("hour", 0))
        minute = int(temporal.get("minute", 0))
        ts_ms = int(temporal.get("timestamp", 0))

        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        current_date_str = dt.strftime("%Y-%m-%d")

        # Safety: derive weekday from the same timestamp when the client value
        # is invalid. Normal client values are preserved.
        if weekday < 0 or weekday > 6:
            weekday = dt.weekday()
        if hour < 0 or hour > 23:
            hour = dt.hour
        if minute < 0 or minute > 59:
            minute = dt.minute

        st = TRAIN_STATS[asset]
        norm_seq = (raw_seq - st["mean"][None, :]) / st["std"][None, :]
        norm_seq = np.clip(norm_seq, -8, 8).astype(np.float32)

        ctx_vec, context_ready = build_live_ctx(
            asset=asset,
            weekday=weekday,
            hour=hour,
            minute=minute,
            current_date_str=current_date_str,
            live_seq=raw_seq,
        )

        date_ordinal = dt.toordinal()
        meta_np = np.array(
            [[0, asset_id, weekday, hour, minute, date_ordinal]],
            dtype=np.int32,
        )

        x_t = torch.from_numpy(norm_seq).unsqueeze(0).to(DEVICE)
        ctx_t = torch.from_numpy(ctx_vec).unsqueeze(0).to(DEVICE)
        meta_t = torch.from_numpy(meta_np).to(DEVICE)

        with torch.no_grad():
            logits = model(x_t, ctx_t, meta_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_down = float(probs[0])
        p_up = float(probs[1])
        total = p_down + p_up
        p_down /= total
        p_up /= total

        inference_ms = int((time.time() - t0) * 1000)

        log.info(
            "PREDICT asset=%s wd=%d hr=%d min=%d | p_up=%.4f p_down=%.4f ctx=%s %dms",
            asset, weekday, hour, minute, p_up, p_down,
            "ready" if context_ready else "prior", inference_ms,
        )

        return jsonify({
            "success": True,
            "model_version": MODEL_VERSION,
            "p_down": round(p_down, 6),
            "p_up": round(p_up, 6),
            "context_ready": context_ready,
            "inference_ms": inference_ms,
        })

    except Exception:
        log.error("Inference error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
