#!/bin/bash
# Publish this folder to a new public GitHub repo, safely.
# Usage (from inside the project folder):  ./publish_to_github.sh <repo-name> "<one-line description>"
set -e
NAME="$1"; DESC="$2"
[ -z "$NAME" ] && { echo "Usage: ./publish_to_github.sh <repo-name> \"<description>\""; exit 1; }
command -v gh >/dev/null || { echo "GitHub CLI not installed. Run: brew install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not signed in. Run: gh auth login"; exit 1; }

[ -d .git ] || git init -b main -q
LOGIN=$(gh api user --jq .login); ID=$(gh api user --jq .id)
git config user.name "$(gh api user --jq '.name // .login')"
git config user.email "${ID}+${LOGIN}@users.noreply.github.com"   # keeps your real email out of the public history

git add -A
echo "Files that will be published:"; git diff --cached --name-only | sed 's/^/  /'

# Safety checks: stop if anything secret or personal is about to be published.
BAD=$(git diff --cached --name-only | grep -E '(^|/)\.env$|jobs\.db|(^|/)config\.yaml$|(^|/)resume\.txt$|resume_.*\.txt|applications/|logs/|backup_|\.zip$|\.venv/|\.bin/|^results/|^downloads\.txt$|\.(m4a|mp3|mp4|wav|m3u8)$|data/(audio|press_releases|cache)/[^.]' || true)
if [ -n "$BAD" ]; then echo; echo "STOPPED: these private files would be published:"; echo "$BAD"; git reset -q; exit 1; fi
KEYS=$(git grep --cached -nI -E 'sk-ant-[A-Za-z0-9_-]{10,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[bp]-|APCA-API-SECRET-KEY *[:=] *[A-Za-z0-9]{10,}' || true)
if [ -n "$KEYS" ]; then echo; echo "STOPPED: something that looks like an API key was found:"; echo "$KEYS" | cut -c1-80; git reset -q; exit 1; fi

echo; read -p "Everything above looks safe. Publish as PUBLIC repo '$NAME'? (y/n) " ok
[ "$ok" = "y" ] || { echo "Cancelled. Nothing was published."; git reset -q; exit 0; }
git commit -q -m "Initial commit"
gh repo create "$NAME" --public --source=. --remote=origin --push --description "$DESC"
echo; echo "Published: https://github.com/$LOGIN/$NAME"
