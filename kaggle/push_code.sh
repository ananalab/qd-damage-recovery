#!/usr/bin/env bash
# Upload the project code to Kaggle as a PRIVATE dataset (anaisazouaoui/qd-damage-recovery-code).
# Usage: bash kaggle/push_code.sh [version message]
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=$(mktemp -d)
COMMIT=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs pyproject.toml || echo "-dirty")
echo "$COMMIT" > CODE_COMMIT
# Exact local versions, without pytinyrenderer (not installable on Kaggle, see kernel_utils.install).
grep -v -i -E "^(pytinyrenderer|appnope)==" requirements-lock.txt > requirements-kaggle.txt
# COPYFILE_DISABLE / --no-mac-metadata: otherwise macOS tar adds "._*" files (Apple metadata).
# On Linux, "src/._qd_damage.egg-info" looks like a package and breaks pip.
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --exclude='._*' --exclude='*.egg-info' --exclude='__pycache__' \
  -czf "$BUILD/qd_damage_code.tar.gz" CODE_COMMIT requirements-kaggle.txt pyproject.toml README.md src configs tests kaggle/kernel_utils.py
python3 -c "import sys,tarfile; bad=[n for n in tarfile.open(sys.argv[1]).getnames() if '._' in n or 'egg-info' in n]; sys.exit(f'unclean archive: {bad}' if bad else 0)" "$BUILD/qd_damage_code.tar.gz"
rm CODE_COMMIT requirements-kaggle.txt
cat > "$BUILD/dataset-metadata.json" <<EOF
{"title": "qd-damage-recovery-code", "id": "anaisazouaoui/qd-damage-recovery-code", "licenses": [{"name": "other"}]}
EOF

if .venv/bin/kaggle datasets status anaisazouaoui/qd-damage-recovery-code >/dev/null 2>&1; then
  .venv/bin/kaggle datasets version -p "$BUILD" -m "${1:-code $COMMIT}"
else
  .venv/bin/kaggle datasets create -p "$BUILD"   # private by default
fi
echo "code $COMMIT uploaded"
