"""Settings, paths and the list of calls to backtest.

Most settings can be overridden in the .env file at the project root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional for the demo
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
AUDIO_DIR = DATA / "audio"
PRESS_RELEASE_DIR = DATA / "press_releases"
CACHE_DIR = DATA / "cache"
RESULTS_DIR = ROOT / "results"
CALLS_CSV = ROOT / "calls.csv"


def _env(name: str, default=None):
    value = os.getenv(name)
    return default if value in (None, "") else value


# --- API keys (live in .env, never in code) ---
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
ALPACA_API_KEY = _env("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _env("ALPACA_SECRET_KEY")

# --- Models ---
JUDGE_MODEL = _env("JUDGE_MODEL", "claude-sonnet-5-5")
WHISPER_MODEL = _env("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# --- Trading / scoring assumptions ---
MIN_CONFIDENCE = float(_env("MIN_CONFIDENCE", 0.70))      # threshold used for the chart
# Pre-registered tests (locked Nov 8, 2026, before the big batch). Change only with a written reason.
TEST_A_CONFIDENCE = 0.70   # official test, judged on the 15-minute hold
TEST_B_CONFIDENCE = 0.55   # secondary test
DEDUPE_MIN = float(_env("DEDUPE_MIN", 5))  # same-direction signals this close together count as one trade
COST_BPS_PER_SIDE = float(_env("COST_BPS_PER_SIDE", 10))  # after-hours spread + slippage, each way
HORIZONS_MIN = [1, 5, 15]                                  # how long each trade is held

# --- Simulated delays (seconds), so the backtest isn't unrealistically fast ---
TRANSCRIBE_LATENCY_SEC = float(_env("TRANSCRIBE_LATENCY_SEC", 0.5))  # live speech-to-text lag
ORDER_LATENCY_SEC = float(_env("ORDER_LATENCY_SEC", 1.0))            # sending the order
# The judge's own response time is measured on every call and added on top.

# --- Fast/slow layer behaviour ---
TRIGGER_THRESHOLD = float(_env("TRIGGER_THRESHOLD", 1.0))  # fast-layer score needed to wake the judge
HEARTBEAT_SEC = float(_env("HEARTBEAT_SEC", 90))           # also ask the judge at least this often
CONTEXT_SEC = float(_env("CONTEXT_SEC", 120))              # how much recent transcript the judge sees
MAX_JUDGE_CALLS = int(_env("MAX_JUDGE_CALLS", 150))        # cost cap per earnings call
ANONYMIZE = _env("ANONYMIZE", "true").lower() == "true"    # hide company name from the judge
# Calls often start a little after their scheduled time. Unless a call's start was verified, assume it
# began this many seconds late. Errs on the side of making the bot look WORSE, never better.
START_SLACK_SEC = float(_env("START_SLACK_SEC", 60))


@dataclass
class Call:
    call_id: str
    ticker: str
    company_name: str
    call_start_et: str          # e.g. "2026-10-28 17:00"
    audio_file: str             # file name inside data/audio/
    audio_offset_sec: float     # seconds into the audio when the call actually starts
    press_release_file: str     # file name inside data/press_releases/ (optional)
    eps_actual: float | None
    eps_estimate: float | None
    revenue_actual: float | None
    revenue_estimate: float | None
    notes: str = ""
    expectations: str = ""      # what analysts expect for NEXT quarter's guidance, in plain words
    start_verified: bool = False  # True if call_start_et was measured exactly (no slack added)

    @property
    def call_start_utc(self) -> pd.Timestamp:
        start = pd.Timestamp(self.call_start_et).tz_localize("America/New_York").tz_convert("UTC")
        return start if self.start_verified else start + pd.Timedelta(seconds=START_SLACK_SEC)

    @property
    def audio_path(self) -> Path:
        return AUDIO_DIR / self.audio_file

    def press_release_text(self) -> str:
        if not self.press_release_file:
            return ""
        path = PRESS_RELEASE_DIR / self.press_release_file
        return path.read_text(errors="ignore") if path.exists() else ""


def _num(value):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "").replace("$", ""))
    except ValueError:
        return None


def load_calls(path: Path = CALLS_CSV) -> dict[str, Call]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    calls = {}
    for _, r in df.iterrows():
        if not r.get("call_id", "").strip():
            continue
        calls[r["call_id"].strip()] = Call(
            call_id=r["call_id"].strip(),
            ticker=r["ticker"].strip().upper(),
            company_name=r.get("company_name", "").strip(),
            call_start_et=r["call_start_et"].strip(),
            audio_file=r["audio_file"].strip(),
            audio_offset_sec=_num(r.get("audio_offset_sec")) or 0.0,
            press_release_file=r.get("press_release_file", "").strip(),
            eps_actual=_num(r.get("eps_actual")),
            eps_estimate=_num(r.get("eps_estimate")),
            revenue_actual=_num(r.get("revenue_actual")),
            revenue_estimate=_num(r.get("revenue_estimate")),
            notes=r.get("notes", "").strip(),
            expectations=r.get("expectations", "").strip(),
            start_verified=r.get("start_verified", "").strip().lower() in ("yes", "true", "1"),
        )
    return calls
