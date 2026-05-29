#!/usr/bin/env python3
"""
Train a logistic regression model to predict next-day stock direction.

Features per sample (computed from historical closes up to day t):
  rsi, macd_bull, above_ma50, above_ma200, bb_pos, volatility, ret5, vol_delta

Label: 1 if close[t+1] > close[t], else 0

Usage:
  TD_API_KEY=<key> python train.py
  python train.py --api-key <key>
"""

import argparse
import math
import os
import sys
import time

import httpx
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOGL", "JPM"]

FEATURE_NAMES = ["rsi", "macd_bull", "above_ma50", "above_ma200",
                 "bb_pos", "volatility", "ret5", "vol_delta"]


# ── Indicator helpers (self-contained, no import from main.py) ────────────────

def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    ch = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    ag = sum(max(c, 0) for c in ch[:period]) / period
    al = sum(max(-c, 0) for c in ch[:period]) / period
    for c in ch[period:]:
        ag = (ag * (period - 1) + max(c, 0)) / period
        al = (al * (period - 1) + max(-c, 0)) / period
    return 50.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _macd_bull(closes):
    if len(closes) < 26:
        return 0
    k12, k26, k9 = 2/13, 2/27, 2/10
    e12 = e26 = closes[0]
    ml = []
    for i in range(1, len(closes)):
        e12 = closes[i] * k12 + e12 * (1 - k12)
        e26 = closes[i] * k26 + e26 * (1 - k26)
        if i >= 25:
            ml.append(e12 - e26)
    if not ml:
        return 0
    sig = ml[0]
    for m in ml[1:]:
        sig = m * k9 + sig * (1 - k9)
    return int(ml[-1] > sig)


def _sma(closes, period):
    return sum(closes[-period:]) / period if len(closes) >= period else None


def _bb_pos(closes, period=20):
    if len(closes) < period:
        return 50.0
    sl = closes[-period:]
    mean = sum(sl) / period
    std = math.sqrt(sum((x - mean) ** 2 for x in sl) / period)
    if std == 0:
        return 50.0
    return max(0.0, min(100.0, ((closes[-1] - (mean - 2 * std)) / (4 * std)) * 100))


def _volatility(closes):
    if len(closes) < 6:
        return 0.013
    rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
    if not rets:
        return 0.013
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
    return vol if vol > 0 else 0.013


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list[str], api_key: str) -> dict:
    sym = ",".join(tickers)
    resp = httpx.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": sym, "interval": "1day", "outputsize": 252, "apikey": api_key},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


# ── Feature/label construction ────────────────────────────────────────────────

def build_dataset(tickers: list[str], api_key: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"Fetching {len(tickers)} tickers from Twelve Data…")
    raw = fetch_ohlcv(tickers, api_key)
    is_single = len(tickers) == 1

    X: list[list[float]] = []
    y: list[int] = []

    for ticker in tickers:
        d = raw if is_single else raw.get(ticker, {})
        if not d.get("values") or d.get("status") == "error" or d.get("code"):
            print(f"  {ticker}: no data — skipping")
            continue

        vals    = list(reversed(d["values"]))
        closes  = [float(v["close"])         for v in vals]
        volumes = [float(v.get("volume", 0)) for v in vals]

        # Need SMA200 history + 1 future day for label → start at index 200
        samples, up_days = 0, 0
        for t in range(200, len(closes) - 1):
            c   = closes[:t + 1]
            vol = volumes[:t + 1]

            sma50    = _sma(c, 50)
            sma200   = _sma(c, 200)
            avg_vol  = sum(vol[-20:]) / max(1, min(20, len(vol)))

            row = [
                _rsi(c),
                _macd_bull(c),
                int(sma50  is not None and c[-1] > sma50),
                int(sma200 is not None and c[-1] > sma200),
                _bb_pos(c),
                _volatility(c[-30:]),
                (c[-1] - c[-6]) / c[-6] * 100 if len(c) > 5 else 0.0,
                (vol[-1] / avg_vol - 1) * 100 if avg_vol else 0.0,
            ]
            label = int(closes[t + 1] > closes[t])
            X.append(row)
            y.append(label)
            samples  += 1
            up_days  += label

        print(f"  {ticker}: {samples} samples — {up_days}/{samples} up-days "
              f"({up_days/samples:.0%})")

    return np.array(X), np.array(y)


# ── Training ──────────────────────────────────────────────────────────────────

def train(api_key: str) -> None:
    X, y = build_dataset(TICKERS, api_key)
    print(f"\nDataset: {len(X)} samples | base rate (up-days): {y.mean():.2%}\n")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Baseline (always-up): {y.mean():.3f}")

    model.fit(X, y)

    coefs = model.named_steps["clf"].coef_[0]
    print("\nFeature coefficients (sorted by importance):")
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True):
        bar = "#" * int(abs(coef) * 20)
        print(f"  {name:15s} {coef:+.3f}  {bar}")

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump({
        "model":       model,
        "features":    FEATURE_NAMES,
        "cv_accuracy": float(scores.mean()),
        "cv_std":      float(scores.std()),
        "tickers":     TICKERS,
    }, out)
    print(f"\nSaved -> {out}")
    print(f"Model accuracy: {scores.mean():.1%} (5-fold CV on {len(X)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("TD_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: provide TD_API_KEY env var or --api-key flag")
        sys.exit(1)

    train(args.api_key)
