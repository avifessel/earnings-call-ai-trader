"""Turn a call recording into a list of words with exact timestamps.

Uses Whisper running locally on your Mac (mlx-whisper, fast on Apple silicon).
Results are cached, so each recording is only transcribed once.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import config


def _ensure_ffmpeg() -> None:
    """Whisper needs ffmpeg to read audio. Use the copy bundled with imageio-ffmpeg if none is installed."""
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError("ffmpeg not found. Re-run ./setup.sh, or install it with: brew install ffmpeg") from e
    bin_dir = config.ROOT / ".bin"
    bin_dir.mkdir(exist_ok=True)
    link = bin_dir / "ffmpeg"
    if not link.exists():
        link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def transcribe(audio_path: Path, cache_path: Path) -> list[dict]:
    """Return [{"w": word, "start": sec, "end": sec}, ...] for the whole recording."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _ensure_ffmpeg()
    print(f"  Transcribing {audio_path.name} (first run downloads the Whisper model, ~1.5 GB)...")
    words: list[dict] = []
    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=config.WHISPER_MODEL,
            word_timestamps=True,
            language="en",
            verbose=False,
        )
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({"w": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    except ImportError:
        # Non-Mac fallback
        from faster_whisper import WhisperModel

        model = WhisperModel(os.getenv("FASTER_WHISPER_MODEL", "large-v3-turbo"), compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language="en")
        for seg in segments:
            for w in seg.words or []:
                words.append({"w": w.word.strip(), "start": float(w.start), "end": float(w.end)})

    words = [w for w in words if w["w"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(words))
    print(f"  Transcribed {len(words):,} words ({words[-1]['end'] / 60:.0f} min of audio)." if words else "  No speech found.")
    return words
