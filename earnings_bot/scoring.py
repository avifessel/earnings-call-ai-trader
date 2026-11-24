"""Score signals against what the price actually did, after trading costs.

Two pre-registered tests (decided before seeing the big batch, so they can't be tuned to results):
  Test A (official): confidence >= TEST_A_CONFIDENCE (0.70), judged on the 15-minute hold
  Test B (secondary): confidence >= TEST_B_CONFIDENCE (0.55)
Everything else here (other holding times, smarter exits) is exploration.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .config import Call

FIXED = [f"hold_{h}m" for h in config.HORIZONS_MIN]
EXITS = ["trailing_stop", "stop_or_target", "opposite_signal", "call_end"]
ALL_RULES = FIXED + EXITS

TRAIL_PCT = 0.5        # trailing stop: exit once price gives back 0.5% from its best point
STOP_PCT = 0.5         # stop-loss at -0.5% ...
TARGET_PCT = 1.0       # ... or take profit at +1.0%
MAX_HOLD_MIN = 30      # every exploratory exit closes by this many minutes, whatever happens


def price_at(bars: pd.DataFrame, ts: pd.Timestamp, max_gap_min: int = 10) -> float:
    """Price you could realistically trade at after time `ts`: the open of the next minute bar.
    (Conservative: a decision at 17:03:20 fills at the 17:04 bar's open.)"""
    later = bars[bars.index > ts.floor("min")]
    if later.empty or (later.index[0] - ts) > pd.Timedelta(minutes=max_gap_min):
        return np.nan
    return float(later["open"].iloc[0])


def _net(direction: int, entry: float, exit_: float) -> float:
    if not entry or np.isnan(entry) or exit_ is None or np.isnan(exit_):
        return np.nan
    return direction * (exit_ / entry - 1) * 1e4 - 2 * config.COST_BPS_PER_SIDE


def select_trades(signals: list[dict], min_conf: float) -> list[dict]:
    """Signals that become trades. A same-direction signal within DEDUPE_MIN of the last trade in that
    direction is the same news, not a new trade, so it's skipped (e.g. two AMD remarks 14s apart)."""
    trades, last = [], {}
    for s in sorted(signals, key=lambda x: x["decided_at"]):
        if s["signal"] == "neutral" or s["confidence"] < min_conf:
            continue
        t = pd.Timestamp(s["decided_at"])
        prev = last.get(s["signal"])
        if prev is not None and t - prev < pd.Timedelta(minutes=config.DEDUPE_MIN):
            continue
        last[s["signal"]] = t
        trades.append(s)
    return trades


def _exit_by_path(bars, direction, entry_time, entry, rule, stop_time=None):
    """Walk minute bars after entry; exit at the next bar's open once the rule triggers."""
    end = entry_time + pd.Timedelta(minutes=MAX_HOLD_MIN)
    if stop_time is not None:
        end = min(end, stop_time)
    path = bars[(bars.index > entry_time.floor("min")) & (bars.index <= end)]
    if path.empty:
        return np.nan
    best = entry
    for ts, row in path.iterrows():
        close = float(row["close"])
        move = direction * (close / entry - 1) * 100
        if rule == "trailing_stop":
            best = max(best, close) if direction == 1 else min(best, close)
            giveback = direction * (best - close) / entry * 100
            if giveback >= TRAIL_PCT:
                return price_at(bars, ts)
        elif rule == "stop_or_target":
            if move <= -STOP_PCT or move >= TARGET_PCT:
                return price_at(bars, ts)
    return price_at(bars, end) if not np.isnan(price_at(bars, end)) else float(path["close"].iloc[-1])


def score_trades(trades: list[dict], bars: pd.DataFrame, all_signals: list[dict], min_conf: float,
                 call_end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for s in trades:
        direction = 1 if s["signal"] == "bullish" else -1
        t = pd.Timestamp(s["decided_at"])
        entry = price_at(bars, t)
        r = {"call_clock": s["call_clock"], "signal": s["signal"], "confidence": s["confidence"],
             "quote": s["quote"], "reason": s["reason"], "decided_at": s["decided_at"], "entry_price": entry}
        for h in config.HORIZONS_MIN:
            r[f"hold_{h}m"] = _net(direction, entry, price_at(bars, t + pd.Timedelta(minutes=h)))
        r["trailing_stop"] = _net(direction, entry, _exit_by_path(bars, direction, t, entry, "trailing_stop"))
        r["stop_or_target"] = _net(direction, entry, _exit_by_path(bars, direction, t, entry, "stop_or_target"))
        opposite = "bearish" if direction == 1 else "bullish"
        opp_times = [pd.Timestamp(x["decided_at"]) for x in all_signals
                     if x["signal"] == opposite and x["confidence"] >= min_conf and pd.Timestamp(x["decided_at"]) > t]
        stop_at = min(opp_times) if opp_times else None
        r["opposite_signal"] = _net(direction, entry, _exit_by_path(bars, direction, t, entry, "none", stop_at))
        r["call_end"] = _net(direction, entry, price_at(bars, max(call_end, t + pd.Timedelta(minutes=1))))
        rows.append(r)
    return pd.DataFrame(rows, columns=["call_clock", "signal", "confidence", "quote", "reason", "decided_at",
                                       "entry_price"] + ALL_RULES)


def baseline(call: Call, bars: pd.DataFrame) -> dict | None:
    """The dumb strategy to beat: at call start, buy if EPS beat estimates, short if it missed."""
    if call.eps_actual is None or call.eps_estimate is None or call.eps_actual == call.eps_estimate:
        return None
    direction = 1 if call.eps_actual > call.eps_estimate else -1
    entry_time = call.call_start_utc + pd.Timedelta(seconds=config.ORDER_LATENCY_SEC)
    entry = price_at(bars, entry_time)
    out = {"direction": "long" if direction == 1 else "short"}
    for h in config.HORIZONS_MIN:
        out[f"hold_{h}m"] = _net(direction, entry, price_at(bars, entry_time + pd.Timedelta(minutes=h)))
    return out


def stats(values) -> dict:
    """Per-trade stats vs. doing nothing (0). t-stat > ~2 is the usual bar for 'probably not luck'."""
    v = pd.Series(values, dtype=float).dropna()
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "win": None, "t": None}
    sd = v.std(ddof=1) if n > 1 else float("nan")
    t = v.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None
    return {"n": n, "mean": round(v.mean(), 1), "median": round(v.median(), 1),
            "win": round((v > 0).mean(), 2), "t": round(t, 2) if t is not None else None}
