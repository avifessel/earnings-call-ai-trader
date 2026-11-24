"""Replay a past call as if it were live: the bot only 'hears' what had been said up to each moment."""
from __future__ import annotations

import pandas as pd

from . import config, triggers


def _clock(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def run_replay(words: list[dict], judge, call_start_utc: pd.Timestamp, audio_offset_sec: float,
               max_minutes: float | None = None, verbose: bool = True) -> list[dict]:
    words = [w for w in words if w["end"] >= audio_offset_sec]
    phrases = triggers.segment(words)
    signals: list[dict] = []
    last_judged = audio_offset_sec
    n_calls = 0

    for ph in phrases:
        call_sec = ph.end - audio_offset_sec
        if max_minutes is not None and call_sec > max_minutes * 60:
            break
        trig_score, labels = triggers.score(ph.text)
        triggered = abs(trig_score) >= config.TRIGGER_THRESHOLD
        heartbeat = ph.end - last_judged >= config.HEARTBEAT_SEC
        if not (triggered or heartbeat):
            continue
        if n_calls >= config.MAX_JUDGE_CALLS:
            if verbose:
                print(f"  Hit MAX_JUDGE_CALLS ({config.MAX_JUDGE_CALLS}); stopping early.")
            break

        # Only words that had finished being spoken by now (no peeking ahead)
        recent = " ".join(w["w"] for w in words if ph.end - config.CONTEXT_SEC <= w["end"] <= ph.end)
        verdict = judge.judge(recent, ph.text, _clock(call_sec))
        n_calls += 1
        last_judged = ph.end

        heard_at = call_start_utc + pd.Timedelta(seconds=call_sec)
        decided_at = heard_at + pd.Timedelta(
            seconds=config.TRANSCRIBE_LATENCY_SEC + verdict["latency_sec"] + config.ORDER_LATENCY_SEC
        )
        sig = {
            "call_clock": _clock(call_sec),
            "call_sec": round(call_sec, 2),
            "heard_at": heard_at.isoformat(),
            "decided_at": decided_at.isoformat(),
            "why_judged": ", ".join(labels) if triggered else "heartbeat",
            "trigger_score": round(trig_score, 2),
            "phrase": ph.text,
            **verdict,
        }
        signals.append(sig)
        if verbose and sig["signal"] != "neutral":
            print(f"  [{sig['call_clock']}] {sig['signal'].upper():8s} conf {sig['confidence']:.2f}  \"{sig['quote'][:90]}\"")

    if verbose:
        print(f"  Judge consulted {n_calls} times over {len(phrases)} phrases.")
    return signals
