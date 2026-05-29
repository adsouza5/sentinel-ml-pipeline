import base64
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery, firestore, pubsub_v1, secretmanager
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Structured JSON logging (Cloud Logging picks this up automatically) ───────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "severity": record.levelname,
            "message":  record.getMessage(),
            "logger":   record.name,
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
logger = logging.getLogger("pipeline")

# ─────────────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="Market Data Signal Pipeline API", version="3.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://iadamdsouza.com",
        "https://www.iadamdsouza.com",
        "https://adsouza5.github.io",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

PROJECT_ID = os.environ.get("GCP_PROJECT", "portfolio-ml-pipeline")
TOPIC_ID   = "market-data-ingest"

_td_key_cache: str | None = None
_publisher:    pubsub_v1.PublisherClient | None = None
_fs_client:    firestore.Client | None = None
_model:        Any | None = None


def get_td_key() -> str:
    global _td_key_cache
    if _td_key_cache:
        return _td_key_cache
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/td-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    _td_key_cache = response.payload.data.decode("UTF-8").strip()
    logger.info("TD API key loaded from Secret Manager")
    return _td_key_cache


def get_publisher() -> pubsub_v1.PublisherClient:
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


def get_model() -> dict | None:
    global _model
    if _model is not None:
        return _model
    path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(path):
        logger.warning("model.joblib not found — falling back to rule-based scoring")
        return None
    _model = joblib.load(path)
    logger.info("ML model loaded cv_accuracy=%.3f", _model.get("cv_accuracy", 0))
    return _model


def get_firestore() -> firestore.Client:
    global _fs_client
    if _fs_client is None:
        _fs_client = firestore.Client(project=PROJECT_ID)
    return _fs_client


# ── Firestore session helpers ─────────────────────────────────────────────────

def session_create(session_id: str) -> None:
    get_firestore().collection("sessions").document(session_id).set({
        "results":    [],
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("session_create session_id=%s", session_id)


def session_append(session_id: str, row: dict) -> None:
    get_firestore().collection("sessions").document(session_id).update({
        "results": firestore.ArrayUnion([row])
    })
    logger.info("session_append session_id=%s ticker=%s", session_id, row.get("ticker"))


def session_get(session_id: str) -> list[dict]:
    doc = get_firestore().collection("sessions").document(session_id).get()
    if not doc.exists:
        return []
    return doc.to_dict().get("results", [])


# ── Indicator math ────────────────────────────────────────────────────────────

def calc_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    ag = sum(max(c, 0) for c in changes[:period]) / period
    al = sum(max(-c, 0) for c in changes[:period]) / period
    for c in changes[period:]:
        ag = (ag * (period - 1) + max(c, 0)) / period
        al = (al * (period - 1) + max(-c, 0)) / period
    return 50.0 if al == 0 else round(100 - 100 / (1 + ag / al), 1)


def calc_macd_bull(closes: list[float]) -> bool:
    if len(closes) < 26:
        return False
    k12, k26, k9 = 2 / 13, 2 / 27, 2 / 10
    e12 = e26 = closes[0]
    ml: list[float] = []
    for i in range(1, len(closes)):
        e12 = closes[i] * k12 + e12 * (1 - k12)
        e26 = closes[i] * k26 + e26 * (1 - k26)
        if i >= 25:
            ml.append(e12 - e26)
    if not ml:
        return False
    sig = ml[0]
    for m in ml[1:]:
        sig = m * k9 + sig * (1 - k9)
    return ml[-1] > sig


def calc_sma(closes: list[float], period: int) -> float | None:
    return sum(closes[-period:]) / period if len(closes) >= period else None


def calc_bollinger_position(closes: list[float], period: int = 20) -> float:
    if len(closes) < period:
        return 50.0
    sl = closes[-period:]
    mean = sum(sl) / period
    std = math.sqrt(sum((x - mean) ** 2 for x in sl) / period)
    if std == 0:
        return 50.0
    pos = ((closes[-1] - (mean - 2 * std)) / (4 * std)) * 100
    return round(max(0.0, min(100.0, pos)), 1)


def calc_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    if len(highs) < 2:
        return 0.0
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(highs))
    ]
    sl = trs[-period:]
    return round(sum(sl) / len(sl), 2)


def calc_volatility(closes: list[float]) -> float:
    if len(closes) < 6:
        return 0.013
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not rets:
        return 0.013
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
    return round(vol, 6) if vol > 0 else 0.013


def score_predict_ml(
    rsi: float, macd_bull: bool, above50: bool, above200: bool,
    bb_pos: float, volatility: float, ret5: float, vol_delta: float,
) -> tuple[str, float]:
    """ML scorer — falls back to rule-based if model unavailable."""
    artifact = get_model()
    if artifact is None:
        return score_predict(rsi, macd_bull, above50, above200, ret5)
    try:
        model    = artifact["model"]
        features = np.array([[rsi, int(macd_bull), int(above50), int(above200),
                               bb_pos, volatility, ret5, vol_delta]])
        prob_up  = float(model.predict_proba(features)[0][1])
        if prob_up >= 0.55:
            return "BULLISH", round(min(0.95, prob_up), 2)
        if prob_up <= 0.45:
            return "BEARISH", round(min(0.95, 1.0 - prob_up), 2)
        return "NEUTRAL", round(max(prob_up, 1.0 - prob_up), 2)
    except Exception as e:
        logger.warning("ml_score_failed falling_back reason=%s", e)
        return score_predict(rsi, macd_bull, above50, above200, ret5)


def score_predict(rsi: float, macd_bull: bool, above50: bool, above200: bool, ret5: float) -> tuple[str, float]:
    score = 0.0
    score += 2 if rsi < 30 else -2 if rsi > 70 else 1 if rsi > 55 else -1 if rsi < 45 else 0
    score += 1.5 if macd_bull else -1.5
    score += 1 if above50 else -1
    score += 0.5 if above200 else -0.5
    score += 0.5 if ret5 > 2 else -0.5 if ret5 < -2 else 0
    norm = score / 5.5
    if norm > 0.2:
        return "BULLISH", round(min(0.95, 0.55 + norm * 0.4), 2)
    if norm < -0.2:
        return "BEARISH", round(min(0.95, 0.55 + abs(norm) * 0.4), 2)
    return "NEUTRAL", round(min(0.78, 0.55 + (0.2 - abs(norm)) * 1.5), 2)


async def fetch_and_score(tickers: list[str]) -> dict[str, dict]:
    """Batch-fetch OHLCV for all tickers in one API call, compute all indicators."""
    td_key = get_td_key()
    sym = ",".join(tickers)

    logger.info("fetch_and_score tickers=%s", sym)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": sym, "interval": "1day", "outputsize": 252, "apikey": td_key},
        )
    fetch_ms = (time.perf_counter() - t0) * 1000
    logger.info("twelve_data_fetch_ms=%.0f status=%d", fetch_ms, resp.status_code)

    if resp.status_code != 200:
        raise HTTPException(502, "Market data fetch failed")

    ts = resp.json()
    is_single = len(tickers) == 1
    results: dict[str, dict] = {}

    for ticker in tickers:
        d = ts if is_single else ts.get(ticker, {})
        if not d.get("values") or d.get("status") == "error" or d.get("code"):
            logger.warning("no_data ticker=%s code=%s", ticker, d.get("code"))
            continue

        vals    = list(reversed(d["values"]))
        closes  = [float(v["close"])         for v in vals]
        highs   = [float(v["high"])          for v in vals]
        lows    = [float(v["low"])           for v in vals]
        volumes = [float(v.get("volume", 0)) for v in vals]

        price  = closes[-1]
        volume = volumes[-1]

        rsi        = calc_rsi(closes)
        macd_bull  = calc_macd_bull(closes)
        sma50      = calc_sma(closes, 50)
        sma200     = calc_sma(closes, 200)
        above50    = sma50  is not None and price > sma50
        above200   = sma200 is not None and price > sma200
        bb_pos     = calc_bollinger_position(closes)
        atr        = calc_atr(highs, lows, closes)
        volatility = calc_volatility(closes)

        avg_vol   = sum(volumes[-20:]) / max(1, min(20, len(volumes)))
        vol_delta = round((volume / avg_vol - 1) * 100, 1) if avg_vol else 0.0
        ret5      = (price - closes[-6]) / closes[-6] * 100 if len(closes) > 5 else 0.0
        sentiment = max(-99, min(99, round(ret5 * 6)))

        prediction, confidence = score_predict_ml(
            rsi, macd_bull, above50, above200, bb_pos, volatility, ret5, vol_delta
        )
        latency = max(10, round(fetch_ms / len(tickers)))

        results[ticker] = {
            "ticker":      ticker,
            "price":       round(price, 2),
            "volume":      int(volume),
            "rsi":         rsi,
            "macd_bull":   macd_bull,
            "above_ma50":  above50,
            "above_ma200": above200,
            "bb_pos":      bb_pos,
            "atr":         atr,
            "volatility":  volatility,
            "vol_delta":   vol_delta,
            "sentiment":   sentiment,
            "prediction":  prediction,
            "confidence":  confidence,
            "latency":     latency,
        }
        logger.info("scored ticker=%s prediction=%s confidence=%.2f rsi=%.1f",
                    ticker, prediction, confidence, rsi)

    return results


def write_to_bq(row: dict) -> None:
    try:
        bq = bigquery.Client(project=PROJECT_ID)
        bq.insert_rows_json(f"{PROJECT_ID}.ml_pipeline.predictions", [{
            "ticker":      row["ticker"],
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "price":       row["price"],
            "rsi":         row["rsi"],
            "macd_bull":   row["macd_bull"],
            "above_ma50":  row["above_ma50"],
            "above_ma200": row["above_ma200"],
            "bb_pos":      row["bb_pos"],
            "atr":         row["atr"],
            "vol_delta":   row["vol_delta"],
            "sentiment":   row["sentiment"],
            "prediction":  row["prediction"],
            "confidence":  row["confidence"],
            "latency_ms":  row["latency"],
        }])
        logger.info("bq_write ticker=%s prediction=%s", row["ticker"], row["prediction"])
    except Exception as e:
        logger.error("bq_write_failed ticker=%s error=%s", row.get("ticker"), e)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    tickers: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "project": PROJECT_ID, "version": "3.0.0"}


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, req: PredictRequest):
    """Synchronous batch prediction — used for custom ticker lookup."""
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    if not tickers:
        raise HTTPException(400, "No tickers provided")
    if len(tickers) > 10:
        raise HTTPException(400, "Max 10 tickers per request")

    logger.info("predict tickers=%s", tickers)
    t0 = time.perf_counter()
    results_map = await fetch_and_score(tickers)
    fetch_ms = (time.perf_counter() - t0) * 1000

    for row in results_map.values():
        write_to_bq(row)

    return {"results": list(results_map.values()), "fetch_ms": round(fetch_ms)}


@app.post("/predict/async")
@limiter.limit("10/minute")
async def predict_async(request: Request, req: PredictRequest):
    """
    Fetch all ticker data in one batch, then publish one Pub/Sub message per
    ticker carrying the computed result. The push subscriber writes each result
    to Firestore (session store) and BigQuery independently.
    """
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    if not tickers:
        raise HTTPException(400, "No tickers provided")
    if len(tickers) > 10:
        raise HTTPException(400, "Max 10 tickers per request")

    logger.info("predict_async tickers=%s", tickers)
    results_map = await fetch_and_score(tickers)

    session_id = str(uuid.uuid4())
    session_create(session_id)

    publisher  = get_publisher()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    message_ids = []

    for ticker in tickers:
        row = results_map.get(ticker)
        if not row:
            continue
        data = json.dumps({
            "session_id": session_id,
            "ticker":     ticker,
            "result":     row,
        }).encode("utf-8")
        future = publisher.publish(topic_path, data)
        message_ids.append(future.result())

    logger.info("predict_async session_id=%s published=%d", session_id, len(message_ids))
    return {
        "session_id":   session_id,
        "ticker_count": len(message_ids),
        "message_ids":  message_ids,
    }


@app.post("/pubsub/push")
async def pubsub_push(request: Request):
    """
    Pub/Sub push delivery. Receives a pre-computed result payload, stores it
    in Firestore, writes to BigQuery. Returns 2xx to acknowledge.
    """
    try:
        body       = await request.json()
        raw        = base64.b64decode(body["message"]["data"]).decode("utf-8")
        payload    = json.loads(raw)
        session_id = payload["session_id"]
        row        = payload["result"]
    except Exception as e:
        logger.warning("pubsub_push invalid message: %s", e)
        return {"status": "invalid_message"}

    try:
        session_append(session_id, row)
    except Exception as e:
        logger.error("session_append failed session_id=%s error=%s", session_id, e)

    write_to_bq(row)
    return {"status": "ok", "ticker": row.get("ticker")}


@app.get("/results/{session_id}")
def get_results(session_id: str):
    """Poll endpoint — returns results collected so far for this session."""
    results = session_get(session_id)
    return {"results": results, "count": len(results)}


@app.get("/analytics")
def analytics():
    """Query BigQuery for prediction statistics over the last 7 days."""
    try:
        bq = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT
                prediction,
                COUNT(*)                    AS count,
                ROUND(AVG(confidence), 3)   AS avg_confidence,
                ROUND(AVG(rsi), 1)          AS avg_rsi,
                COUNT(DISTINCT ticker)      AS unique_tickers
            FROM `{PROJECT_ID}.ml_pipeline.predictions`
            WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            GROUP BY prediction
            ORDER BY count DESC
        """
        rows = list(bq.query(query).result())
        total = sum(int(r["count"]) for r in rows)
        logger.info("analytics query total_predictions=%d", total)
        return {
            "total_predictions": total,
            "window_days":       7,
            "breakdown": [
                {
                    "prediction":     r["prediction"],
                    "count":          int(r["count"]),
                    "avg_confidence": float(r["avg_confidence"] or 0),
                    "avg_rsi":        float(r["avg_rsi"] or 0),
                    "unique_tickers": int(r["unique_tickers"]),
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.error("analytics_query_failed error=%s", e)
        return {"total_predictions": 0, "window_days": 7, "breakdown": []}
