"""The FAST layer: split speech into phrases and flag the ones worth a closer look.

This runs in microseconds and costs nothing. It doesn't decide trades; it only decides
when to wake up the slower, smarter judge. Positive weights lean bullish, negative bearish,
but only the size of the score matters for triggering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Phrase:
    text: str
    start: float
    end: float


def segment(words: list[dict], pause_sec: float = 0.6, max_words: int = 40) -> list[Phrase]:
    """Group words into phrases: end at sentence punctuation, a pause, or a length cap."""
    phrases, buf = [], []
    for i, w in enumerate(words):
        buf.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        ends_sentence = w["w"].endswith((".", "?", "!"))
        long_pause = nxt is not None and nxt["start"] - w["end"] >= pause_sec
        if ends_sentence or long_pause or len(buf) >= max_words or nxt is None:
            phrases.append(Phrase(" ".join(x["w"] for x in buf), buf[0]["start"], buf[-1]["end"]))
            buf = []
    return phrases


# (pattern, weight, label)
_RULES = [
    # Guidance changes: the most market-moving thing said on most calls
    (r"\b(rais(e|es|ed|ing)|increas(e|es|ed|ing)|lift(ed|ing)?)\b.{0,40}\b(guidance|outlook|forecast)", 2.0, "guidance up"),
    (r"\b(lower(s|ed|ing)?|cut(s|ting)?|reduc(e|es|ed|ing)|trim(med|ming)?)\b.{0,40}\b(guidance|outlook|forecast)", -2.0, "guidance down"),
    (r"\b(reaffirm(s|ed|ing)?|maintain(s|ed|ing)?|reiterat(e|es|ed|ing))\b.{0,40}\b(guidance|outlook)", 0.8, "guidance held"),
    (r"\b(withdraw(s|n|ing)?|suspend(s|ed|ing)?|pull(s|ed|ing)?)\b.{0,40}\b(guidance|outlook)", -2.5, "guidance pulled"),
    # Strong positive language
    (r"\brecord\b", 0.8, "record"),
    (r"\b(accelerat(e|es|ed|ing|ion))\b", 1.0, "acceleration"),
    (r"\b(exceed(s|ed|ing)?|outperform(s|ed|ing)?|ahead of (plan|expectations))\b", 1.0, "beat language"),
    (r"\bstrong(er|est)? (demand|growth|momentum|bookings|pipeline)\b", 1.0, "strong demand"),
    (r"\bmargin expansion\b|\bexpand(ed|ing)? (our )?margins?\b", 1.0, "margin up"),
    (r"\b(buyback|repurchase|dividend increase|raising (the|our) dividend)\b", 1.0, "capital return"),
    # Negative language
    (r"\bheadwinds?\b", -1.0, "headwind"),
    (r"\b(decelerat(e|es|ed|ing|ion)|slow(ed|ing|down)|softness|soft(er)?)\b", -1.0, "slowdown"),
    (r"\b(below (our )?expectations|fell short|disappoint(ing|ed)?|miss(ed)?)\b", -1.2, "miss language"),
    (r"\bmargin (pressure|compression)\b|\bpressure on (our )?margins?\b", -1.0, "margin down"),
    (r"\b(uncertain(ty)?|cautious|challenging|difficult (environment|quarter))\b", -0.8, "caution"),
    (r"\b(restructuring|layoffs?|impairment|write[- ]?(down|off)|investigation|recall)\b", -1.2, "bad event"),
    (r"\b(pause|pausing|delay(ed|ing)?|push(ed)? out)\b", -0.8, "delay"),
    (r"\b(supply constraint|shortage|inventory (build|correction|digestion))\b", -0.8, "supply/inventory"),
]
_COMPILED = [(re.compile(p, re.I), w, label) for p, w, label in _RULES]

_FORWARD = re.compile(r"\b(guidance|outlook|expect|anticipate|forecast|full[- ]year|next quarter|fiscal (20)?\d\d|for the (year|quarter))\b", re.I)
_NUMBER = re.compile(r"(\d|\bpercent\b|\bbillion\b|\bmillion\b|\bbasis points\b)", re.I)


def score(text: str) -> tuple[float, list[str]]:
    """Return (signed score, labels that fired)."""
    total, labels = 0.0, []
    for rx, weight, label in _COMPILED:
        if rx.search(text):
            total += weight
            labels.append(label)
    if _FORWARD.search(text) and _NUMBER.search(text):
        labels.append("forward-looking number")
        total += 1.0 if total >= 0 else -1.0  # makes it stronger in whichever direction it leans
    return total, labels
