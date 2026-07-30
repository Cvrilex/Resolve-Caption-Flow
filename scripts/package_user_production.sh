#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
PACKAGE_NAME="ResolveCaptionFlow-user"
OUTPUT_DIR="$(cd "$REPO_ROOT/.." && pwd)"
INCLUDE_DOCS=0
KEEP_TEMP=0

usage() {
  cat <<'EOF'
Usage: scripts/package_user_production.sh [options]

Build a clean user production zip. The package keeps application code,
resources, launcher scripts, requirements, and reusable knowledge-base JSON
files. It excludes tests, vendor references, git metadata, virtualenvs, logs,
videos, generated SRTs, intermediate work files, and local WebUI secrets.

Options:
  --output-dir DIR     Write the zip to DIR. Default: parent of repo root.
  --name NAME          Package folder name inside the zip. Default: ResolveCaptionFlow-user.
  --include-docs       Include docs/ in the package.
  --keep-temp          Keep the temporary staging folder for inspection.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      PACKAGE_NAME="$2"
      shift 2
      ;;
    --include-docs)
      INCLUDE_DOCS=1
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$REPO_ROOT/app" || ! -f "$REPO_ROOT/start_web.command" ]]; then
  echo "This script must be run from the Resolve Caption Flow repository layout." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
ZIP_PATH="$OUTPUT_DIR/${PACKAGE_NAME}-production-${STAMP}.zip"
STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/resolve-caption-flow-package.XXXXXX")"
APP_DIR="$STAGING_ROOT/$PACKAGE_NAME"

cleanup() {
  if [[ "$KEEP_TEMP" -eq 0 ]]; then
    rm -rf "$STAGING_ROOT"
  else
    echo "Temporary staging folder kept at: $STAGING_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$APP_DIR"

copy_dir() {
  local source="$1"
  local target="$2"
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$source" "$target"
}

copy_dir "$REPO_ROOT/app" "$APP_DIR/"
copy_dir "$REPO_ROOT/caption_core" "$APP_DIR/"
copy_dir "$REPO_ROOT/integrations" "$APP_DIR/"
copy_dir "$REPO_ROOT/pipeline" "$APP_DIR/"
copy_dir "$REPO_ROOT/resources" "$APP_DIR/"

if [[ "$INCLUDE_DOCS" -eq 1 ]]; then
  copy_dir "$REPO_ROOT/docs" "$APP_DIR/"
fi

cp "$REPO_ROOT/README.md" "$APP_DIR/README.md"
cp "$REPO_ROOT/requirements.txt" "$APP_DIR/requirements.txt"
cp "$REPO_ROOT/start_web.command" "$APP_DIR/start_web.command"
chmod +x "$APP_DIR/start_web.command"

mkdir -p \
  "$APP_DIR/data/input" \
  "$APP_DIR/data/output" \
  "$APP_DIR/data/logs" \
  "$APP_DIR/data/work"

touch \
  "$APP_DIR/data/input/.gitkeep" \
  "$APP_DIR/data/output/.gitkeep" \
  "$APP_DIR/data/logs/.gitkeep" \
  "$APP_DIR/data/work/.gitkeep"

find "$APP_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$APP_DIR" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
rm -f "$APP_DIR/data/work/web_config.json"

cat > "$APP_DIR/PACKAGE_INFO.txt" <<EOF
Resolve Caption Flow user production package
Built at: $(date '+%Y-%m-%d %H:%M:%S %z')
Source: $REPO_ROOT

Included:
- app/, caption_core/, integrations/, pipeline/, resources/
- README.md, requirements.txt, start_web.command

Excluded:
- .git, tests, vendor references, virtual environments
- input videos, output videos, logs, generated SRTs, intermediate work files
- local WebUI config, API keys, and all local knowledge-base data

Start:
- Double-click start_web.command on macOS.
EOF

(
  cd "$STAGING_ROOT"
  zip -qr "$ZIP_PATH" "$PACKAGE_NAME"
)

SHA256="$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')"
SIZE="$(du -h "$ZIP_PATH" | awk '{print $1}')"

cat <<EOF
Production package created:
$ZIP_PATH

Size: $SIZE
SHA256: $SHA256
EOF
