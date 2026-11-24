#!/usr/bin/env python3
"""Backtest the earnings-call bot on past calls.

  python run.py --demo                 # fake data, checks everything works (free)
  python run.py --check                # checks your API keys and setup
  python run.py AAPL_2026Q3            # backtest one call from calls.csv
  python run.py --all                  # backtest every call in calls.csv
  python run.py AAPL_2026Q3 --offline-judge --max-minutes 10   # cheap dry run

Results go to results/<call_id>/ (chart.html, signals.csv, trades.csv) and results/summary.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser

import pandas as pd

from earnings_bot import config
from earnings_bot.chart import make_chart
from earnings_bot.judge import ClaudeJudge, LexiconJudge, build_context
from earnings_bot.prices import get_minute_bars, last_regular_close
from earnings_bot.replay import run_replay
from earnings_bot.scoring import ALL_RULES, EXITS, FIXED, baseline, price_at, score_trades, select_trades, stats


def backtest(call, args, words=None, bars=None):
    print(f"\n=== {call.call_id} ({call.ticker}) ===")
    out_dir = config.RESULTS_DIR / call.call_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Transcript with word timestamps
    if words is None:
        from earnings_bot.transcribe import transcribe
        words = transcribe(call.audio_path, config.CACHE_DIR / f"{call.call_id}_words.json")
    if not words:
        print("  No transcript; skipping.")
        return None
    call_start = call.call_start_utc
    call_end = call_start + pd.Timedelta(seconds=words[-1]["end"] - call.audio_offset_sec)

    # 2. Prices: from the prior morning through 45 min after the call
    if bars is None:
        bars = get_minute_bars(call.ticker, call_start - pd.Timedelta(hours=20),
                               call_end + pd.Timedelta(minutes=45),
                               config.CACHE_DIR / f"{call.call_id}_bars.csv")
    close_before = last_regular_close(bars, call_start)
    at_start = price_at(bars, call_start)
    release_move = (at_start / close_before - 1) * 100 if close_before and pd.notna(at_start) else None
    if release_move is not None:
        print(f"  Stock already moved {release_move:+.2f}% on the press release before the call started.")

    # 3. Replay the call through the fast/slow layers (cached, since the AI judge costs money)
    judge_name = "lexicon" if args.offline_judge else "claude"
    sig_path = out_dir / f"signals_{judge_name}.json"
    if sig_path.exists() and not args.rejudge:
        signals = json.loads(sig_path.read_text())
        print(f"  Using saved {judge_name} signals ({len(signals)}). Use --rejudge to redo them.")
    else:
        if args.offline_judge:
            judge = LexiconJudge()
        else:
            judge = ClaudeJudge(call, build_context(call, release_move))
        signals = run_replay(words, judge, call_start, call.audio_offset_sec, max_minutes=args.max_minutes)
        sig_path.write_text(json.dumps(signals, indent=1))
        if judge.input_tokens:
            print(f"  Tokens used: {judge.input_tokens:,} in / {judge.output_tokens:,} out.")
    # Re-time saved verdicts from the current call start (so timing fixes never require re-paying Claude).
    for sg in signals:
        heard = call_start + pd.Timedelta(seconds=sg["call_sec"])
        sg["heard_at"] = heard.isoformat()
        sg["decided_at"] = (heard + pd.Timedelta(seconds=config.TRANSCRIBE_LATENCY_SEC + sg["latency_sec"]
                                                  + config.ORDER_LATENCY_SEC)).isoformat()
    pd.DataFrame(signals).to_csv(out_dir / "signals.csv", index=False)

    # 4. Score each pre-registered test vs. what the price actually did, and the dumb baseline
    tests = {"A": config.TEST_A_CONFIDENCE, "B": config.TEST_B_CONFIDENCE}
    if args.min_confidence is not None:
        tests["Custom"] = args.min_confidence
    trades = {}
    for name, thr in tests.items():
        df = score_trades(select_trades(signals, thr), bars, signals, thr, call_end)
        df.insert(0, "call_id", call.call_id)
        df.insert(1, "test", name)
        trades[name] = df
        df.to_csv(out_dir / f"trades_{name}.csv", index=False)
    base = baseline(call, bars)

    # 5. Chart (shows Test A's threshold)
    chart_path = out_dir / "chart.html"
    make_chart(call.call_id, call.ticker, bars, call_start, call_end, signals, config.TEST_A_CONFIDENCE, chart_path)

    def avg(df, col):
        v = df[col].dropna()
        return f"{v.mean():+.0f} bps" if len(v) else "n/a"
    for name, thr in tests.items():
        df = trades[name]
        print(f"  Test {name} (conf >= {thr:.2f}): {len(df)} trades"
              + (f" | 15m hold avg {avg(df, 'hold_15m')}" if len(df) else ""))
    if base:
        print(f"  EPS baseline ({base['direction']}): 15m hold {base['hold_15m']:+.0f} bps")
    print(f"  Chart: {chart_path}")
    if args.open:
        webbrowser.open(chart_path.as_uri())
    row = {"call_id": call.call_id, "release_move_pct": round(release_move, 2) if release_move is not None else None,
           "judge_calls": len(signals), **{f"trades_{n}": len(df) for n, df in trades.items()},
           **{f"baseline_{r}": (round(base[r], 1) if base and not pd.isna(base[r]) else None) for r in FIXED}}
    return {"row": row, "trades": trades, "baseline": base}


def print_summary(results, tests):
    n_calls = len(results)
    print(f"\n=== Results across {n_calls} call(s): net bps per trade after costs (100 bps = 1%) ===")
    print("Compared with doing nothing (0). 't' above ~2 means 'probably not luck'; below that, could easily be luck.")
    for name, thr in tests.items():
        df = pd.concat([r["trades"][name] for r in results], ignore_index=True)
        calls_with = df["call_id"].nunique() if len(df) else 0
        label = "official" if name == "A" else ("secondary" if name == "B" else "custom")
        print(f"\nTest {name} ({label}, confidence >= {thr:.2f}): {len(df)} trades in {calls_with} calls")
        print(f"  {'exit rule':<18}{'trades':>7}{'avg':>8}{'median':>8}{'win%':>7}{'t':>7}")
        for rule in ALL_RULES:
            if rule == EXITS[0]:
                print("  -- exploration only --")
            st = stats(df[rule]) if len(df) else stats([])
            mark = "  <- official" if (name == "A" and rule == "hold_15m") else ""
            fmt = lambda v, f: (f.format(v) if v is not None else "-")
            print(f"  {rule:<18}{st['n']:>7}{fmt(st['mean'], '{:+.1f}'):>8}{fmt(st['median'], '{:+.1f}'):>8}"
                  f"{fmt(st['win'] * 100 if st['win'] is not None else None, '{:.0f}'):>7}{fmt(st['t'], '{:.2f}'):>7}{mark}")
    bases = [r["baseline"] for r in results if r["baseline"]]
    print(f"\nEPS baseline (one trade per call, {len(bases)} calls):")
    for rule in FIXED:
        st = stats([b[rule] for b in bases])
        print(f"  {rule:<18}{st['n']:>7}{(format(st['mean'], '+.1f') if st['mean'] is not None else '-'):>8}")


def check_setup():
    ok = True
    print("Checking setup...")
    for name in ("ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        present = bool(getattr(config, name))
        ok &= present
        print(f"  {name:20s} {'found' if present else 'MISSING (add to .env)'}")
    try:
        import mlx_whisper  # noqa: F401
        print("  Whisper (mlx)        installed")
    except ImportError:
        print("  Whisper (mlx)        NOT installed (run ./setup.sh)")
        ok = False
    if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
        try:
            now = pd.Timestamp.now(tz="UTC")
            df = get_minute_bars("AAPL", now - pd.Timedelta(days=5), now - pd.Timedelta(hours=1),
                                 config.CACHE_DIR / "_check_bars.csv")
            (config.CACHE_DIR / "_check_bars.csv").unlink(missing_ok=True)
            print(f"  Alpaca data          working ({len(df)} AAPL minute bars)")
        except Exception as e:
            ok = False
            print(f"  Alpaca data          FAILED: {e}")
    if config.ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            client.messages.create(model=config.JUDGE_MODEL, max_tokens=5,
                                   messages=[{"role": "user", "content": "Say ok"}])
            print(f"  Anthropic API        working ({config.JUDGE_MODEL})")
        except Exception as e:
            ok = False
            print(f"  Anthropic API        FAILED: {e}")
    print("All good!" if ok else "Fix the items above, then run this again.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("call_ids", nargs="*", help="call_id values from calls.csv")
    p.add_argument("--all", action="store_true", help="backtest every call in calls.csv")
    p.add_argument("--demo", action="store_true", help="run on built-in fake data")
    p.add_argument("--check", action="store_true", help="check API keys and installs")
    p.add_argument("--offline-judge", action="store_true", help="use free keyword judge instead of Claude")
    p.add_argument("--rejudge", action="store_true", help="ignore saved signals and re-run the judge")
    p.add_argument("--max-minutes", type=float, help="only replay the first N minutes (cheaper tests)")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="also score an extra 'Custom' threshold (exploration; Tests A and B always run)")
    p.add_argument("--open", action="store_true", help="open each chart in your browser")
    args = p.parse_args()

    if args.check:
        return check_setup()

    rows = []
    if args.demo:
        from earnings_bot import demo
        args.offline_judge = True
        args.rejudge = True
        call = demo.demo_call()
        rows.append(backtest(call, args, words=demo.demo_words(), bars=demo.demo_bars(call)))
    else:
        calls = config.load_calls()
        ids = list(calls) if args.all else args.call_ids
        if not ids:
            p.print_help()
            sys.exit("\nNo calls given. Add rows to calls.csv, then pass a call_id or --all.")
        for cid in ids:
            if cid not in calls:
                print(f"\n'{cid}' isn't in calls.csv; skipping.")
                continue
            try:
                rows.append(backtest(calls[cid], args))
            except Exception as e:
                print(f"  ERROR on {cid}: {e}")

    results = [r for r in rows if r]
    if not results:
        return
    tests = {"A": config.TEST_A_CONFIDENCE, "B": config.TEST_B_CONFIDENCE}
    if args.min_confidence is not None:
        tests["Custom"] = args.min_confidence
    suffix = "_demo" if args.demo else ""
    pd.DataFrame([r["row"] for r in results]).to_csv(config.RESULTS_DIR / f"summary{suffix}.csv", index=False)
    pd.concat([df for r in results for df in r["trades"].values()], ignore_index=True).to_csv(
        config.RESULTS_DIR / f"all_trades{suffix}.csv", index=False)
    print_summary(results, tests)
    print(f"\nSaved: results/summary{suffix}.csv (per call) and results/all_trades{suffix}.csv (every trade)")


if __name__ == "__main__":
    main()
