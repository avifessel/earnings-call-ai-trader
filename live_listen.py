#!/usr/bin/env python3
"""Listen to a LIVE earnings-call webcast, transcribe it as it happens, and measure how far
behind the real call we are. This is a feasibility test: it does NOT trade.

  python live_listen.py "<live stream link>" mchp_live

Get the stream link the same way as for replays (Inspect > Network, filter m3u8, copy the row),
but while the call is live. Stop any time with Ctrl+C.

Saves:  data/audio/<name>.m4a         (the recording, usable later for backtesting)
        results/live/<name>.txt       (timestamped transcript + delay measurements)
"""
from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from earnings_bot import config
from grab_audio import UA, ffmpeg_path

SR = 16000
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- broadcast delay
def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=10).read().decode(errors="ignore")


def broadcast_delay(url: str) -> tuple[float | None, str]:
    """Estimate how many seconds the webcast runs behind the real call, using the timing tags
    (EXT-X-PROGRAM-DATE-TIME) that most live streams include. Returns (seconds, note)."""
    try:
        text = _get(url)
        if "#EXT-X-STREAM-INF" in text:  # a master playlist: follow its first stream
            first = next(l for l in text.splitlines() if l and not l.startswith("#"))
            url = urllib.parse.urljoin(url, first)
            text = _get(url)
        if "#EXT-X-ENDLIST" in text:
            return None, "this is a finished recording, not a live stream"
        t = edge = None
        dur = 0.0
        durs = []
        for line in text.splitlines():
            if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                t = datetime.fromisoformat(line.split(":", 1)[1].strip().replace("Z", "+00:00")).timestamp()
            elif line.startswith("#EXTINF:"):
                dur = float(line.split(":", 1)[1].split(",")[0])
                durs.append(dur)
            elif line and not line.startswith("#") and t is not None:
                edge = t + dur
                t = edge
        if edge is None or not durs:
            return None, "stream has no timing tags, so the delay can't be measured this way"
        newest_age = time.time() - edge          # how old the newest available audio is
        buffer = 3 * max(durs)                   # players (and ffmpeg) start ~3 segments back
        return newest_age + buffer, f"{max(durs):.0f}s segments"
    except Exception as e:  # never let this stop the listener
        return None, f"couldn't read stream timing ({str(e)[:60]})"


# ---------------------------------------------------------------- transcription
class Transcriber:
    def __init__(self, model: str):
        self.model = model
        try:
            import mlx_whisper
            self._mlx = mlx_whisper
        except ImportError:
            sys.exit("mlx-whisper isn't installed. Run ./setup.sh first.")
        # Warm up (loads the model) and time one step so we know if it can keep up.
        self.text(np.zeros(SR, dtype=np.float32), "")
        t0 = time.time()
        self.text((np.random.randn(5 * SR) * 0.01).astype(np.float32), "")
        self.step_sec = time.time() - t0

    def text(self, audio: np.ndarray, prompt: str) -> str:
        out = self._mlx.transcribe(
            audio, path_or_hf_repo=self.model, language="en",
            initial_prompt=prompt or None, condition_on_previous_text=False, verbose=None,
            temperature=0.0,  # no slow retries
        )
        # Drop Whisper "hallucinations" (e.g. on hold music): repetitive or low-confidence text.
        segs = [s for s in out.get("segments", [])
                if s.get("no_speech_prob", 0) < 0.6
                and s.get("compression_ratio", 0) <= 2.2
                and s.get("avg_logprob", 0) > -1.0]
        text = " ".join(s["text"].strip() for s in segs).strip()
        words = text.lower().split()
        if len(words) >= 8 and len(set(words)) / len(words) < 0.3:  # e.g. "a man, a man, a man..."
            return ""
        return text


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", help="live stream link (.m3u8) copied from the browser (omit with --mic)")
    p.add_argument("name", nargs="?", help="name for the saved files, e.g. mchp_live")
    p.add_argument("--mic", action="store_true",
                   help="record what the Mac hears through its microphone instead of a stream link "
                        "(works with any webcast player: play the call in your browser out loud)")
    p.add_argument("--device", help="audio input device name or number for --mic (see --list-devices)")
    p.add_argument("--list-devices", action="store_true", help="show audio input devices and exit")
    p.add_argument("--chunk", type=float, default=5.0, help="seconds of audio per transcription step (default 5)")
    p.add_argument("--model", default="mlx-community/whisper-base.en-mlx",
                   help="Whisper model for live use. Default is small and fast; "
                        "try mlx-community/whisper-small.en-mlx for better accuracy if your Mac keeps up")
    p.add_argument("--referer", help="the webcast page's address, if the stream refuses to load")
    args = p.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.mic:
        if args.name is None:
            args.name, args.url = args.url, None
        if not args.name:
            sys.exit("Give a name, e.g.:  python live_listen.py --mic nvda_live")
        if args.device is not None and str(args.device).isdigit():
            args.device = int(args.device)
    else:
        if not args.url or not args.name:
            sys.exit("Usage: python live_listen.py \"<stream link>\" name   (or: --mic name)")
        if not args.url.startswith(("http://", "https://")):
            sys.exit("That's not a web link. Copy the row whose Name ends in .m3u8.")

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = config.RESULTS_DIR / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = config.AUDIO_DIR / f"{args.name}.m4a"
    log_path = out_dir / f"{args.name}.txt"
    log = open(log_path, "a", buffering=1)

    def say(line: str):
        print(line)
        log.write(line + "\n")

    if args.mic:
        delay, note = None, "unknown in microphone mode (the browser player's own delay, usually 10-30s, applies)"
    else:
        delay, note = broadcast_delay(args.url)
    say(f"=== {args.name} | started {datetime.now(ET):%Y-%m-%d %H:%M:%S} ET ===")
    say(f"Webcast delay behind the real call: " + (f"~{delay:.0f}s ({note})" if delay is not None else note))

    print(f"Loading Whisper ({args.model}; first time downloads it)...")
    asr = Transcriber(args.model)
    say(f"Transcription speed: {asr.step_sec:.1f}s per step "
        + ("(OK, keeps up)" if asr.step_sec < args.chunk * 0.7 else
           f"(TOO SLOW for {args.chunk:.0f}s chunks; try a smaller --model or a bigger --chunk)"))

    raw_path = audio_out.with_suffix(".pcm")
    raw_file = open(raw_path, "wb")
    chunk_bytes = int(args.chunk * SR) * 2
    q: queue.Queue = queue.Queue()
    received = {"sec": 0.0, "first": None}
    stop = threading.Event()
    req_headers = {"User-Agent": UA, **({"Referer": args.referer} if args.referer else {})}

    def fetch(url: str) -> bytes:
        return urllib.request.urlopen(urllib.request.Request(url, headers=req_headers), timeout=15).read()

    def decode(data: bytes) -> bytes:
        """Turn one downloaded stream piece into raw 16 kHz mono audio."""
        r = subprocess.run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                            "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
                           input=data, capture_output=True)
        return r.stdout

    def reader():
        """Poll the live playlist, download each new piece as it appears, decode it, queue the audio."""
        url = args.url
        seen: set[str] = set()
        first = True
        buf = b""
        while not stop.is_set():
            try:
                text = fetch(url).decode(errors="ignore")
                if "#EXT-X-STREAM-INF" in text:  # master playlist: switch to its first stream
                    url = urllib.parse.urljoin(url, next(l for l in text.splitlines() if l and not l.startswith("#")))
                    continue
                segs, pdt, dur = [], None, 0.0
                for line in text.splitlines():
                    if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                        pdt = datetime.fromisoformat(line.split(":", 1)[1].strip().replace("Z", "+00:00")).timestamp()
                    elif line.startswith("#EXTINF:"):
                        dur = float(line.split(":", 1)[1].split(",")[0])
                    elif line and not line.startswith("#"):
                        segs.append((urllib.parse.urljoin(url, line), pdt, dur))
                        pdt = pdt + dur if pdt is not None else None
                if first:  # start at the live edge, not the beginning of the event
                    for u, _, _ in segs[:-1]:
                        seen.add(u)
                    first = False
                for u, start, d in segs:
                    if u in seen:
                        continue
                    seen.add(u)
                    pcm = decode(fetch(u))
                    if not pcm:
                        continue
                    raw_file.write(pcm)
                    received["sec"] += len(pcm) / 2 / SR
                    if received["first"] is None:
                        received["first"] = time.time()
                    buf += pcm
                    end_stamp = start + d if start is not None else None
                    while len(buf) >= chunk_bytes:
                        q.put((buf[:chunk_bytes], time.time(), end_stamp))
                        buf = buf[chunk_bytes:]
                if "#EXT-X-ENDLIST" in text:
                    q.put(None)
                    return
            except Exception as e:
                print(f"   (stream hiccup: {str(e)[:80]}; retrying)")
            stop.wait(1.0)

    def mic_reader():
        """Record from the microphone (or another input device) in real time."""
        import sounddevice as sd
        buf = bytearray()

        def cb(indata, frames, t, status):
            pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes()
            raw_file.write(pcm)
            received["sec"] += len(pcm) / 2 / SR
            if received["first"] is None:
                received["first"] = time.time()
            buf.extend(pcm)
            while len(buf) >= chunk_bytes:
                q.put((bytes(buf[:chunk_bytes]), time.time(), None))
                del buf[:chunk_bytes]

        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=args.device,
                                blocksize=int(SR * 0.5), callback=cb):
                stop.wait()
        except Exception as e:
            print(f"   Microphone error: {e}\n   (If macOS asked for microphone access, allow it for Terminal, "
                  "then run again.)")
            q.put(None)

    threading.Thread(target=mic_reader if args.mic else reader, daemon=True).start()
    say("Listening. Lines show: ET clock | minutes into stream | text | how long transcription took.\n")

    stream_t = 0.0
    start_wall = time.time()
    prompt = ""
    lags: list[float] = []
    behinds: list[float] = []
    last_delay_check = time.time()
    delays = [delay] if delay is not None else []
    try:
        last_beat = time.time()
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                if received["first"] is None:
                    say(f"   ...no audio received yet ({time.time() - start_wall:.0f}s). Check the link is the live audio stream.")
                else:
                    say(f"   ...waiting for audio (received {received['sec']:.0f}s so far)")
                continue
            if time.time() - last_beat > 30:
                last_beat = time.time()
                say(f"   ...receiving audio ({received['sec']:.0f}s so far)")
            if item is None:
                say("\nStream ended (or the link stopped working).")
                break
            data, arrived, stamp = item
            # Stay live: if chunks piled up, skip the old ones instead of falling further behind.
            skipped = 0
            while q.qsize() > 1:
                nxt = q.get()
                if nxt is None:
                    q.put(None)
                    break
                stream_t += len(data) / 2 / SR
                data, arrived, stamp = nxt
                skipped += 1
            if skipped:
                say(f"   (skipped {skipped * args.chunk:.0f}s of backlog to stay live)")
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            stream_t += len(audio) / SR
            if np.sqrt(np.mean(audio ** 2)) < 0.003:  # silence / hold
                continue
            text = asr.text(audio, prompt)
            done = time.time()
            if not text:
                continue
            lag = done - arrived
            lags.append(lag)
            behind = ""
            if stamp is not None:
                b = done - stamp  # from the webcast server's timestamp of this audio to text ready
                behinds.append(b)
                behind = f", {b:.0f}s behind"
            prompt = text[-200:]
            m, s = divmod(int(stream_t), 60)
            say(f"{datetime.now(ET):%H:%M:%S} | {m:3d}:{s:02d} | {text}   [{lag:.1f}s{behind}]")
            if done - last_delay_check > 60:
                last_delay_check = done
                d, n = broadcast_delay(args.url)
                if d is not None:
                    delays.append(d)
                    say(f"   (webcast delay now ~{d:.0f}s)")
    except KeyboardInterrupt:
        say("\nStopped.")
    finally:
        stop.set()
        time.sleep(0.5)
        raw_file.close()
        if raw_path.exists() and raw_path.stat().st_size > 0:
            subprocess.run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(SR),
                            "-ac", "1", "-i", str(raw_path), "-c:a", "aac", "-b:a", "64k", str(audio_out)])
            raw_path.unlink(missing_ok=True)

    say("\n=== Delay summary ===")
    web = sum(delays) / len(delays) if delays else None
    proc = sum(lags) / len(lags) if lags else None
    say(f"Webcast behind real call:      " + (f"~{web:.0f}s" if web is not None else "unknown (no timing tags)"))
    if proc is not None:
        say(f"Transcription per chunk:        ~{proc:.1f}s on average")
        say(f"Waiting to fill a chunk:        up to {args.chunk:.0f}s")
        if web is not None:
            say(f"=> A sentence reaches the bot roughly {web + proc + args.chunk / 2:.0f}s after it's spoken "
                f"(before Claude's ~2s and the order).")
    if behinds:
        say(f"MEASURED: text was ready on average {sum(behinds) / len(behinds):.0f}s after the webcast server "
            f"stamped that audio (best {min(behinds):.0f}s, worst {max(behinds):.0f}s).")
        say("   Add a few seconds for the trip from the phone bridge to the webcast server.")
    say(f"Recording saved to {audio_out}")
    say(f"Transcript saved to {log_path}")


if __name__ == "__main__":
    main()
