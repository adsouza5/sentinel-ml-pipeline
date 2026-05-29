import math
import pytest
from main import (
    calc_rsi,
    calc_macd_bull,
    calc_sma,
    calc_bollinger_position,
    calc_atr,
    calc_volatility,
    score_predict,
)


# ── calc_rsi ──────────────────────────────────────────────────────────────────

def test_rsi_insufficient_data_returns_50():
    assert calc_rsi([100.0] * 10) == 50.0


def test_rsi_all_losses_returns_zero():
    closes = [100.0 - i for i in range(16)]  # strictly falling
    assert calc_rsi(closes) == 0.0


def test_rsi_mixed_returns_in_range():
    closes = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106,
              105, 107, 106, 108, 107, 109]
    result = calc_rsi(closes)
    assert 0.0 <= result <= 100.0


def test_rsi_overbought_range():
    # Explicit series: 12 up-days of +1.0, 3 down-days of -0.2 → avg_gain >> avg_loss
    closes = [100.0]
    pattern = [1.0, 1.0, 1.0, 1.0, -0.2, 1.0, 1.0, 1.0, 1.0, -0.2,
               1.0, 1.0, 1.0, 1.0, -0.2]
    for d in pattern:
        closes.append(closes[-1] + d)
    result = calc_rsi(closes)
    assert result > 55.0


def test_rsi_oversold_range():
    # Mostly losses → RSI should be well below 50
    closes = [100 - i * 0.5 if i % 5 != 0 else 100 - i * 0.5 + 0.1
              for i in range(16)]
    result = calc_rsi(closes)
    assert result < 45.0


# ── calc_macd_bull ────────────────────────────────────────────────────────────

def test_macd_insufficient_data_returns_false():
    assert calc_macd_bull([100.0] * 25) is False


def test_macd_flat_series_returns_false():
    # Flat prices → EMA12 == EMA26 → MACD line ≈ 0 throughout → not bullish
    assert calc_macd_bull([100.0] * 30) is False


def test_macd_strongly_rising_is_bullish():
    closes = [100 + i * 2 for i in range(40)]
    assert calc_macd_bull(closes) is True


def test_macd_strongly_falling_is_not_bullish():
    closes = [200 - i * 2 for i in range(40)]
    assert calc_macd_bull(closes) is False


def test_macd_returns_bool():
    result = calc_macd_bull([100 + i for i in range(30)])
    assert isinstance(result, bool)


# ── calc_sma ──────────────────────────────────────────────────────────────────

def test_sma_basic():
    assert calc_sma([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx(4.0)


def test_sma_full_series():
    assert calc_sma([10.0, 20.0, 30.0], 3) == pytest.approx(20.0)


def test_sma_insufficient_returns_none():
    assert calc_sma([1.0, 2.0], 5) is None


def test_sma_period_equals_length():
    assert calc_sma([2.0, 4.0, 6.0], 3) == pytest.approx(4.0)


# ── calc_bollinger_position ───────────────────────────────────────────────────

def test_bollinger_insufficient_data_returns_50():
    assert calc_bollinger_position([100.0] * 10) == 50.0


def test_bollinger_flat_series_returns_50():
    assert calc_bollinger_position([100.0] * 20) == 50.0


def test_bollinger_result_in_range():
    import random
    random.seed(42)
    closes = [100 + random.gauss(0, 2) for _ in range(30)]
    result = calc_bollinger_position(closes)
    assert 0.0 <= result <= 100.0


def test_bollinger_high_price_above_50():
    closes = [100.0] * 19 + [110.0]  # price well above mean
    result = calc_bollinger_position(closes)
    assert result > 50.0


def test_bollinger_low_price_below_50():
    closes = [100.0] * 19 + [90.0]  # price well below mean
    result = calc_bollinger_position(closes)
    assert result < 50.0


# ── calc_atr ──────────────────────────────────────────────────────────────────

def test_atr_insufficient_data_returns_zero():
    assert calc_atr([10.0], [9.0], [9.5]) == 0.0


def test_atr_basic_calculation():
    # TR[1]: max(11-9.5, |11-9.5|, |9.5-9.5|) = 1.5
    # TR[2]: max(10.5-9.8, |10.5-10.5|, |9.8-10.5|) = 0.7
    # ATR = (1.5 + 0.7) / 2 = 1.1
    highs  = [10.0, 11.0, 10.5]
    lows   = [9.0,  9.5,  9.8]
    closes = [9.5,  10.5, 10.2]
    assert calc_atr(highs, lows, closes) == pytest.approx(1.1, abs=0.01)


def test_atr_positive():
    highs  = [105.0, 106.0, 104.0]
    lows   = [100.0, 101.0, 100.0]
    closes = [102.0, 103.0, 101.0]
    assert calc_atr(highs, lows, closes) > 0.0


def test_atr_uses_prior_close_in_true_range():
    # Gap-up scenario: high/low range is small but gap from prior close is large
    highs  = [100.0, 110.0]
    lows   = [99.0,  109.0]
    closes = [99.5,  109.5]
    # TR[1] = max(110-109, |110-99.5|, |109-99.5|) = max(1, 10.5, 9.5) = 10.5
    assert calc_atr(highs, lows, closes) == pytest.approx(10.5, abs=0.01)


# ── calc_volatility ───────────────────────────────────────────────────────────

def test_volatility_insufficient_data_returns_default():
    assert calc_volatility([100.0] * 4) == pytest.approx(0.013)


def test_volatility_flat_series_returns_default():
    assert calc_volatility([100.0] * 10) == pytest.approx(0.013)


def test_volatility_positive():
    closes = [100, 102, 99, 103, 101, 105, 98, 104, 100, 106]
    assert calc_volatility(closes) > 0.0


def test_volatility_high_vol_greater_than_low_vol():
    low_vol  = [100 + (i % 2) * 0.1 for i in range(10)]
    high_vol = [100 + (i % 2) * 5.0 for i in range(10)]
    assert calc_volatility(high_vol) > calc_volatility(low_vol)


# ── score_predict ─────────────────────────────────────────────────────────────

def test_score_predict_strong_bull():
    pred, conf = score_predict(rsi=25.0, macd_bull=True, above50=True, above200=True, ret5=3.0)
    assert pred == "BULLISH"
    assert 0.5 <= conf <= 0.95


def test_score_predict_strong_bear():
    pred, conf = score_predict(rsi=75.0, macd_bull=False, above50=False, above200=False, ret5=-3.0)
    assert pred == "BEARISH"
    assert 0.5 <= conf <= 0.95


def test_score_predict_neutral():
    # rsi=50(0) + macd=False(-1.5) + above50(+1) + above200=False(-0.5) + ret5=0(0) = -1.0
    # norm = -1.0/5.5 = -0.18 → within (-0.2, 0.2) → NEUTRAL
    pred, conf = score_predict(rsi=50.0, macd_bull=False, above50=True, above200=False, ret5=0.0)
    assert pred == "NEUTRAL"
    assert conf <= 0.78


def test_score_predict_confidence_bounded():
    for rsi in [20, 50, 80]:
        for macd in [True, False]:
            _, conf = score_predict(rsi, macd, True, True, 0.0)
            assert 0.0 <= conf <= 0.95


def test_score_predict_returns_valid_labels():
    valid = {"BULLISH", "BEARISH", "NEUTRAL"}
    for rsi in [20, 50, 80]:
        pred, _ = score_predict(rsi, True, True, True, 0.0)
        assert pred in valid
