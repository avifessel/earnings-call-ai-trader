"""Made-up call and made-up prices, so you can check everything works before using real data.
Needs no API keys, no audio and no internet."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Call

SCRIPT = [
    # (seconds into call, speaker line)
    (0, "Good afternoon and welcome to the Demo Corp third quarter earnings conference call."),
    (12, "Joining me today are our chief executive officer and our chief financial officer."),
    (60, "We delivered revenue of four point two billion dollars, up eleven percent year over year."),
    (95, "Demand was healthy across most regions and our teams executed well."),
    (180, "Gross margin was forty one percent, consistent with what we discussed last quarter."),
    (300, "Turning to the outlook, we are raising our full year revenue guidance to seventeen point five billion dollars."),
    (330, "We are seeing strong demand and accelerating bookings in our cloud segment."),
    (480, "Operating expenses came in as planned and we continue to invest in research."),
    (720, "In Europe we did see some softness and we expect those headwinds to persist into next quarter."),
    (900, "Let me now turn it over to the operator for questions."),
    (960, "Can you talk about pricing trends in the enterprise business?"),
    (975, "Sure, pricing has been stable and we are pleased with renewal rates."),
    (1200, "On margins, we do expect some margin pressure next quarter of about one hundred basis points from the new factory ramp."),
    (1500, "That concludes our call. Thank you for joining us."),
]
# Fake market reactions (minutes after call start, % move) so the demo has something to score
REACTIONS = [(5.3, +1.6), (12.3, -0.7), (20.3, -1.1)]


def demo_call() -> Call:
    return Call(
        call_id="DEMO", ticker="DEMO", company_name="Demo Corp", call_start_et="2026-10-28 17:00",
        audio_file="", audio_offset_sec=0.0, press_release_file="",
        eps_actual=1.32, eps_estimate=1.25, revenue_actual=4.2e9, revenue_estimate=4.1e9,
        notes="synthetic demo",
    )


def demo_words() -> list[dict]:
    words = []
    for start, line in SCRIPT:
        t = float(start)
        for w in line.split():
            dur = 0.25 + 0.04 * len(w)
            words.append({"w": w, "start": round(t, 2), "end": round(t + dur, 2)})
            t += dur + 0.08
    return words


def demo_bars(call: Call) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    start = call.call_start_utc - pd.Timedelta(minutes=90)
    idx = pd.date_range(start, call.call_start_utc + pd.Timedelta(minutes=90), freq="min")
    mins_from_start = (idx - call.call_start_utc).total_seconds() / 60
    rets = rng.normal(0, 0.0007, len(idx))
    # The release lands at 16:05 ET (55 min before the call): stock jumps ~4%
    rets[np.argmin(np.abs(mins_from_start + 55))] += 0.04
    for minute, pct in REACTIONS:
        i = int(np.argmin(np.abs(mins_from_start - minute)))
        rets[i:i + 3] += pct / 100 / 3
    close = 96 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
                         "close": close, "volume": 1000}, index=idx)
