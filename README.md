# Live Earnings Call Trader

Can an AI listening to a company's earnings call spot what the market hasn't priced in yet, the way an
analyst would? This project tests that idea honestly before any real money gets near it.

Calls are transcribed with Whisper, a free keyword layer flags phrases worth attention (guidance,
margins, demand), and Claude judges each moment as bullish, bearish or neutral with a confidence score.
Every decision is scored against minute-by-minute stock prices from Alpaca, after trading costs, and
compared with a simple baseline: "buy if earnings per share beat estimates".

**Status (November 2026):** backtester and live listener working; testing is in progress.
- Backtested on 12 real calls so far (Apple, Microsoft, Amazon, AMD, Intel and others). Early results are
  too small a sample to judge, by design. The plan is to expand to hundreds of calls before trusting any result.
- Ran the live listener during a real earnings call: transcription stayed about 6-10 seconds behind
  the broadcast, so live paper trading is technically feasible.
- Paper trading only. No real money until both the large backtest and a full season of live paper trading hold up.

**Design choices that keep the test honest**
- Test thresholds were set before the big backtest (confidence of 0.70 or more, judged on a 15-minute hold),
  so results can't be tuned after the fact.
- The company's name and ticker are hidden from the judge, and only calls after the model's training
  cutoff count as clean tests.
- The replay never peeks ahead; trades fill at the next minute's price, with realistic delays and costs.

**Tech:** Python, OpenAI Whisper (local transcription), Claude API, Alpaca market data, pandas, Plotly charts.

Built in fall 2026 with AI-assisted development (Claude). I designed the experiment, chose the rules
and thresholds, collected and verified the call data, and ran the live test.
Not financial advice: this is a research tool.

---

## Backtester guide

Replays past earnings calls as if they were live, lets an AI judge decide when to trade,
and scores every decision against what the stock actually did, after trading costs.

No real money is involved. This only tests on historical data.

## How it works

```
call recording ──► Whisper (on your Mac) ──► words with timestamps
                                                   │
                         FAST layer: keyword triggers on each phrase (free, instant)
                                                   │ flagged phrase, or every 90 s
                         SLOW layer: Claude reads the last 2 min + press release
                                                   │ bullish / bearish / neutral + confidence
                         simulated delays: speech-to-text + Claude's real response time + order
                                                   │
                         score vs. Alpaca minute prices at +1, +5, +15 min, minus costs
                                                   │
                         compare vs. dumb baseline: "buy if EPS beat, short if missed"
```

## One-time setup

In Terminal:

```bash
cd ~/dev/projects/Live_Earnings_Call_Trader
chmod +x setup.sh
./setup.sh
```

This creates a private Python environment in `.venv`, installs everything, and runs a demo on fake data.
Then:

1. **Get your keys**
   - Anthropic: https://console.anthropic.com/settings/keys (separate from your Claude subscription, billed per use; add a few dollars of credit)
   - Alpaca: sign up free at https://alpaca.markets, open the **Paper** account, click **API Keys → Generate**
2. **Put them in `.env`**: `open -e .env` and paste them after the `=` signs. Don't share this file.
3. **Check it all works**: `.venv/bin/python run.py --check`
4. **Look at the demo chart**: `open results/DEMO/chart.html`

Tip: run `source .venv/bin/activate` once per Terminal window, and then you can type `python` instead of `.venv/bin/python`.

## Adding a real call

1. **Audio** → save the call recording (mp3, m4a, wav or mp4) into `data/audio/`.
   Company investor-relations pages usually host replays for a few months.
2. **Press release** (optional but recommended) → copy the text into a `.txt` file in `data/press_releases/`.
3. **Add a row to `calls.csv`** (open it in Numbers/Excel or any text editor):

| column | example | meaning |
|---|---|---|
| call_id | `NVDA_2026Q3` | any unique name |
| ticker | `NVDA` | |
| company_name | `NVIDIA` | used to hide the name from the judge |
| call_start_et | `2026-11-19 17:00` | Eastern time when the call actually began |
| audio_file | `nvda_q3.mp3` | file name in `data/audio/` |
| audio_offset_sec | `42` | seconds into the recording where the call starts (skip hold music). Find it by listening for the operator's first words. |
| press_release_file | `nvda_q3.txt` | file name in `data/press_releases/` |
| eps_actual / eps_estimate | `0.89` / `0.84` | for the baseline |
| revenue_actual / revenue_estimate | `35100000000` / `33200000000` | optional |
| notes | | anything |

**Getting the timing right matters.** `call_start_et` + `audio_offset_sec` is how each second of audio
gets matched to the stock price at that moment. If they're off by a minute, the results are meaningless.

## Running a backtest

```bash
python run.py NVDA_2026Q3 --offline-judge --max-minutes 10   # free dry run: keywords only, first 10 min
python run.py NVDA_2026Q3 --open                             # full run with Claude, opens the chart
python run.py --all                                          # every call in calls.csv + summary table
python run.py --all --min-confidence 0.8                     # re-score at a stricter threshold (free: reuses saved signals)
python run.py NVDA_2026Q3 --rejudge                          # re-run Claude (e.g. after changing the prompt)
```

The first run on a call transcribes it (a few minutes for an hour of audio) and downloads prices; both are cached.
Claude's verdicts are saved too, so re-scoring at different thresholds costs nothing.

**Output** in `results/<call_id>/`:
- `chart.html`: price during the call with every signal marked (hover for the quote and reasoning)
- `signals.csv`: every time the judge was asked, and what it said
- `trades.csv`: each trade taken, with net profit/loss in basis points (100 bps = 1%) at each holding time

and `results/summary.csv` across all calls.

## Reading the results honestly

- **Compare to the baseline.** If the bot doesn't beat "buy if EPS beat", listening to the call isn't adding anything.
- **Use calls from after July 2026.** Claude's training data runs through mid-2026, so on older calls it may
  already "know" how the stock reacted. The company name is hidden by default (`ANONYMIZE=true`), but
  famous calls can still be recognizable. Recent calls are the only clean test.
- **20–30 calls minimum** before drawing conclusions; a handful of calls is mostly luck.
- **Costs are an assumption.** `COST_BPS_PER_SIDE=10` (0.10% each way) is reasonable for big, liquid stocks
  after hours. Smaller stocks can be far worse; raise it for those.
- **Minute bars are coarse.** Entries fill at the next minute's open, which is conservative but imprecise.

## Rough costs

- Whisper, Alpaca data: free.
- Claude: about 40–100 judge calls per earnings call. With prompt caching, expect very roughly
  $0.50–$2 per call on Sonnet, less on Haiku (`JUDGE_MODEL=claude-haiku-5-5` in `.env`).
  `--max-minutes` and `MAX_JUDGE_CALLS` cap spending. Check exact usage in the Anthropic console.

## Files

```
run.py                     command-line entry point
earnings_bot/config.py     settings and calls.csv loading
earnings_bot/transcribe.py Whisper transcription with word timestamps
earnings_bot/prices.py     Alpaca minute bars
earnings_bot/triggers.py   fast layer: phrase splitting + keyword triggers
earnings_bot/judge.py      slow layer: Claude judge (and free keyword judge)
earnings_bot/replay.py     replays the call with no peeking ahead
earnings_bot/scoring.py    trade returns, costs, baseline
earnings_bot/chart.py      interactive chart
earnings_bot/demo.py       fake call for testing
```

Not financial advice. This is a research tool for testing an idea on historical data.
