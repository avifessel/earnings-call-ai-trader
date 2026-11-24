"""Interactive chart: price during the call with every trade signal marked. Opens in any browser."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from . import config


def make_chart(call_id: str, ticker: str, bars: pd.DataFrame, call_start_utc: pd.Timestamp,
               call_end_utc: pd.Timestamp, signals: list[dict], min_conf: float, out_path: Path) -> None:
    window = bars[(bars.index >= call_start_utc - pd.Timedelta(minutes=30)) &
                  (bars.index <= call_end_utc + pd.Timedelta(minutes=30))]
    if window.empty:
        window = bars
    et = window.index.tz_convert("America/New_York").tz_localize(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=et, y=window["close"], mode="lines", name=f"{ticker} price",
                             line=dict(color="#5b6b7f", width=1.6),
                             hovertemplate="%{x|%H:%M}  $%{y:.2f}<extra></extra>"))

    def to_et(ts):
        return pd.Timestamp(ts).tz_convert("America/New_York").tz_localize(None)

    def price_near(ts):
        idx = window.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
        return float(window["close"].iloc[idx]) if idx >= 0 else None

    styles = {"bullish": ("triangle-up", "#1a9e5c"), "bearish": ("triangle-down", "#d64545")}
    for kind, (symbol, color) in styles.items():
        for traded in (True, False):
            pts = [s for s in signals if s["signal"] == kind and (s["confidence"] >= min_conf) == traded]
            if not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[to_et(s["decided_at"]) for s in pts],
                y=[price_near(s["decided_at"]) for s in pts],
                mode="markers",
                name=f"{kind} ({'traded' if traded else 'below threshold'})",
                marker=dict(symbol=symbol, size=[8 + 14 * s["confidence"] for s in pts], color=color,
                            opacity=1.0 if traded else 0.3, line=dict(width=1, color="white")),
                customdata=[[s["call_clock"], s["confidence"], s["quote"][:160], s["reason"][:200]] for s in pts],
                hovertemplate="<b>%{customdata[0]} into call</b> · conf %{customdata[1]:.2f}<br>"
                              "\"%{customdata[2]}\"<br><i>%{customdata[3]}</i><extra></extra>",
            ))

    for ts, label in ((call_start_utc, "call starts"), (call_end_utc, "call ends")):
        x = to_et(ts)
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(dash="dot", color="#999"))
        fig.add_annotation(x=x, y=1, yref="paper", text=label, showarrow=False, yanchor="bottom", font=dict(size=11))

    fig.update_layout(
        title=f"{call_id}: signals vs. price (trades need confidence ≥ {min_conf:.2f}; "
              f"costs {config.COST_BPS_PER_SIDE:g} bps per side)",
        xaxis_title="Time (ET)", yaxis_title="Price ($)", template="plotly_white",
        hovermode="closest", legend=dict(orientation="h", y=-0.18),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True)
