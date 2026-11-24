"""The SLOW layer: an AI judge that reads recent context and decides whether to trade."""
from __future__ import annotations

import json
import re
import time

from . import config, triggers
from .config import Call

SYSTEM_PROMPT = """You are an experienced equity trader listening to a company's earnings call in real time. \
You only know what has been said so far.

The press release came out before the call, and the market has already reacted to its headline numbers. \
Your job is to judge whether the NEW information in the latest remark should move the stock further from here: \
guidance changes, surprising detail behind the numbers, evasive or confident answers, anything the release didn't reveal.

Rules:
- Most remarks are routine. Default to "neutral".
- Use confidence 0.7 or higher only for clear, material, surprising information.
- Don't react to information that was already in the press release.
- Judge only from the transcript and context given, not from anything you may know about this company.

Reply with JSON only, no other text:
{"signal": "bullish" | "bearish" | "neutral", "confidence": 0.0-1.0, "quote": "<short exact quote from the latest remark>", "reason": "<one sentence>"}"""


def anonymize(text: str, call: Call) -> str:
    names = {call.ticker}
    if call.company_name:
        names.add(call.company_name)
        first = re.sub(r"[^\w&-]", "", call.company_name.split()[0])
        if len(first) >= 4:
            names.add(first)
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}('s)?\b", r"the Company\1", text, flags=re.I)
    return text


def build_context(call: Call, release_move_pct: float | None) -> str:
    lines = ["CONTEXT AVAILABLE BEFORE THE CALL"]
    if call.eps_actual is not None and call.eps_estimate is not None:
        lines.append(f"EPS: reported {call.eps_actual:.2f} vs analyst estimate {call.eps_estimate:.2f}")
    if call.revenue_actual is not None and call.revenue_estimate is not None:
        lines.append(f"Revenue: reported {call.revenue_actual:,.0f} vs estimate {call.revenue_estimate:,.0f}")
    if call.expectations:
        lines.append(f"Analyst expectations going into the call: {call.expectations}")
    if release_move_pct is not None:
        lines.append(f"Stock move from regular-session close to call start (reaction to the release): {release_move_pct:+.2f}%")
    pr = call.press_release_text()
    if pr:
        lines.append("\nPRESS RELEASE:\n" + pr[:15000])
    text = "\n".join(lines)
    return anonymize(text, call) if config.ANONYMIZE else text


def _parse(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    signal = str(data.get("signal", "neutral")).lower()
    if signal not in ("bullish", "bearish", "neutral"):
        signal = "neutral"
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"signal": signal, "confidence": conf, "quote": str(data.get("quote", ""))[:300], "reason": str(data.get("reason", ""))[:400]}


class ClaudeJudge:
    name = "claude"

    def __init__(self, call: Call, context: str, model: str = config.JUDGE_MODEL):
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY missing. Add it to your .env file (or use --offline-judge).")
        import anthropic

        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)
        self.call, self.context, self.model = call, context, model
        self.input_tokens = self.output_tokens = 0

    def judge(self, recent: str, latest: str, clock: str) -> dict:
        if config.ANONYMIZE:
            recent, latest = anonymize(recent, self.call), anonymize(latest, self.call)
        user = (
            f"Transcript of the last ~{config.CONTEXT_SEC / 60:.0f} minutes:\n{recent}\n\n"
            f'Latest remark (just finished, {clock} into the call):\n"{latest}"\n\nYour JSON verdict:'
        )
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                # The context is identical on every request, so cache it to cut cost and latency.
                {"type": "text", "text": self.context, "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": user}],
        )
        latency = time.perf_counter() - t0
        self.input_tokens += resp.usage.input_tokens + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0)
        self.output_tokens += resp.usage.output_tokens
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        out = _parse(raw)
        out["latency_sec"] = round(latency, 3)
        return out


class LexiconJudge:
    """Free, offline stand-in for the AI judge: uses the keyword score's direction.
    Useful for testing the pipeline and as a 'dumb keywords' comparison."""

    name = "lexicon"

    def __init__(self, *args, **kwargs):
        self.input_tokens = self.output_tokens = 0

    def judge(self, recent: str, latest: str, clock: str) -> dict:
        s, labels = triggers.score(latest)
        if abs(s) < 1.5:
            return {"signal": "neutral", "confidence": 0.3, "quote": "", "reason": "below keyword threshold", "latency_sec": 0.0}
        return {
            "signal": "bullish" if s > 0 else "bearish",
            "confidence": min(1.0, 0.5 + abs(s) / 6),
            "quote": latest[:200],
            "reason": "keywords: " + ", ".join(labels),
            "latency_sec": 0.0,
        }
