#!/bin/bash
# One-time setup. Run from Terminal:  ./setup.sh
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 not found. Install it from https://www.python.org/downloads/ and re-run ./setup.sh"
  exit 1
fi

VER=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
ARCH=$("$PY" -c 'import platform; print(platform.machine())')
echo "Using $PY (Python $VER, $ARCH)"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and re-run."
  exit 1
fi
if [ "$ARCH" != "arm64" ]; then
  echo "Warning: this Python is an Intel ($ARCH) build, so the fast Mac Whisper won't install."
  echo "Install the Apple-silicon Python from python.org, then re-run ./setup.sh"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
echo "Installing packages (a few minutes the first time)..."
.venv/bin/pip install --quiet -r requirements.txt

mkdir -p data/audio data/press_releases data/cache results
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Open it and paste in your API keys."
fi

echo ""
echo "Running the demo to check everything works..."
.venv/bin/python run.py --demo
echo ""
echo "Setup done. Next steps:"
echo "  1. Put your API keys in .env   (open -e .env)"
echo "  2. Check them:                  .venv/bin/python run.py --check"
echo "  3. See the demo chart:          open results/DEMO/chart.html"
