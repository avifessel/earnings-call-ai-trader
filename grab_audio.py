#!/usr/bin/env python3
"""Save a streamed earnings-call replay as an audio file in data/audio/.

Works with stream links ending in .m3u8 (most webcast players), .mp3, .mp4 or .m4a.

  python grab_audio.py "<stream link>" aapl_q4_2026     # one call
  python grab_audio.py --list downloads.txt             # every call listed in downloads.txt

downloads.txt has one call per line: a name, a space, then the stream link.
  msft_q1_2027 https://....m3u8
Files already downloaded are skipped, so you can re-run it after adding more lines.

How to find the stream link (Chrome):
  1. On the replay page, right-click > Inspect, then open the Network tab.
  2. Type m3u8 in the filter box (if nothing shows up later, try mp3 or mp4).
  3. Press play on the replay.
  4. Right-click the row that appears > Copy > Copy URL.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

from earnings_bot import config

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _frozen_copy(url: str, referer: str | None) -> str | None:
    """A live stream that was never marked finished still lists every piece, but ffmpeg would wait at
    its end forever. Save a copy of the list marked as finished, so it downloads start to end."""
    if ".m3u8" not in url:
        return None
    try:
        hdrs = {"User-Agent": UA, **({"Referer": referer} if referer else {})}
        text = urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=15).read().decode(errors="ignore")
    except Exception:
        return None
    if "#EXT-X-STREAM-INF" in text or "#EXT-X-ENDLIST" in text or "#EXTINF" not in text:
        return None
    lines = [urllib.parse.urljoin(url, l) if l and not l.startswith("#") else l for l in text.splitlines()]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".m3u8", delete=False)
    tmp.write("\n".join(lines) + "\n#EXT-X-ENDLIST\n")
    tmp.close()
    print("  (stream was never marked finished; downloading everything it has so far)")
    return tmp.name


def download(url: str, name: str, referer: str | None = None) -> bool:
    out = config.AUDIO_DIR / (name if name.endswith(".m4a") else name + ".m4a")
    headers = ["-headers", f"Referer: {referer}\r\n"] if referer else []
    frozen = _frozen_copy(url, referer)
    src = ["-protocol_whitelist", "file,http,https,tcp,tls,crypto", "-i", frozen] if frozen else ["-i", url]
    web_opts = [] if frozen else ["-user_agent", UA, *headers]  # those options only apply to web inputs
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-stats", "-y",
           *web_opts, *src,
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k", str(out)]
    print(f"Downloading {out.name} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0 or not out.exists() or out.stat().st_size < 10_000:
        out.unlink(missing_ok=True)
        print(f"  FAILED: {name}. Check the link is the stream (.m3u8/.mp3/.mp4), not the web page.")
        return False
    print(f"  Saved {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", help="stream link copied from the browser")
    p.add_argument("name", nargs="?", help="file name to save as, e.g. aapl_q4_2026")
    p.add_argument("--list", help="text file with one 'name link' per line")
    p.add_argument("--referer", help="the replay page's address, if the download is refused")
    args = p.parse_args()
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        jobs = []
        for line in open(args.list):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"Skipping line (needs a name and a link): {line}")
                continue
            if not parts[1].startswith(("http://", "https://")):
                print(f"Skipping {parts[0]}: that's not a web link (it starts with '{parts[1][:20]}...'). "
                      "Copy the row whose Name ends in .m3u8, .mp4 or .mp3.")
                continue
            jobs.append((parts[0], parts[1]))
        done, failed, skipped = [], [], []
        for name, url in jobs:
            if (config.AUDIO_DIR / (name + ".m4a")).exists():
                skipped.append(name)
                continue
            (done if download(url, name, args.referer) else failed).append(name)
        print(f"\nDownloaded {len(done)}, already had {len(skipped)}, failed {len(failed)}.")
        if failed:
            print("Failed: " + ", ".join(failed) + "  (the link may have expired; copy a fresh one)")
        return

    if not (args.url and args.name):
        p.print_help()
        sys.exit(1)
    if not args.url.startswith(("http://", "https://")):
        sys.exit("That's not a web link. Copy the row whose Name ends in .m3u8, .mp4 or .mp3.")
    if not download(args.url, args.name, args.referer):
        sys.exit("Try again with --referer \"<replay page address>\" if the link is right.")


if __name__ == "__main__":
    main()
