#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="127.0.0.1"
PORT="8742"
URL="http://${HOST}:${PORT}/"
STATUS_URL="${URL}api/system/status"

cd "$PROJECT_DIR" || exit 1

echo
echo "========================================"
echo " Resolve Caption Flow WebUI"
echo "========================================"
echo "项目目录: $PROJECT_DIR"
echo "访问地址: $URL"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "没有找到 python3。请先安装 Python 3，或检查终端 PATH。"
  echo
  read -r -p "按回车键关闭窗口..."
  exit 1
fi

STATUS_JSON="$(curl -fsS "$STATUS_URL" 2>/dev/null || true)"
if [ -n "$STATUS_JSON" ]; then
  STATUS_CHECK="$(
    STATUS_JSON="$STATUS_JSON" PROJECT_DIR="$PROJECT_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

try:
    data = json.loads(os.environ.get("STATUS_JSON", "{}"))
except Exception:
    print("other|端口上已有服务，但无法识别服务状态。")
    raise SystemExit

service = data.get("service") or {}
job = data.get("job") or {}
reported_root = str(service.get("app_root") or "")
project_root = str(Path(os.environ["PROJECT_DIR"]).resolve())
current_mtime = (Path(project_root) / "app" / "web_server.py").stat().st_mtime
reported_mtime = float(service.get("code_mtime") or 0)
job_status = str(job.get("status") or "idle")

if reported_root != project_root:
    print(f"other|端口 8742 正在运行另一个字幕助手服务：{reported_root or '未知目录'}。")
elif abs(reported_mtime - current_mtime) > 0.001:
    print("stale|检测到 Web 服务仍在运行旧代码，需要重启后才能使用当前生产包。")
else:
    print(f"current|Web 服务已经在当前目录运行，任务状态：{job_status}。")
PY
  )"
  STATUS_ACTION="${STATUS_CHECK%%|*}"
  STATUS_MESSAGE="${STATUS_CHECK#*|}"
  if [ "$STATUS_ACTION" = "current" ]; then
    echo "$STATUS_MESSAGE"
    echo "正在打开网页..."
    open "$URL" >/dev/null 2>&1 || true
    echo
    echo "如果浏览器没有自动打开，请手动访问：$URL"
    echo
    read -r -p "按回车键关闭窗口..."
    exit 0
  fi

  echo "$STATUS_MESSAGE"
  if command -v lsof >/dev/null 2>&1; then
    echo
    echo "占用端口的进程："
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
    echo
    read -r -p "按回车尝试关闭旧字幕助手服务并启动当前版本，或直接关闭窗口取消..."
    PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    for PID in $PIDS; do
      CMDLINE="$(ps -p "$PID" -o command= 2>/dev/null || true)"
      if echo "$CMDLINE" | grep -q "app/web_server.py"; then
        kill "$PID" 2>/dev/null || true
      fi
    done
    sleep 1
  fi
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "端口 $PORT 已被其他程序占用，无法启动 Web 服务。"
  echo
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
  echo
  read -r -p "按回车键关闭窗口..."
  exit 1
fi

VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
REQ_FILE="$PROJECT_DIR/requirements.txt"
STAMP_FILE="$VENV_DIR/.requirements-installed"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "正在准备本地 Python 运行环境..."
  python3 -m venv "$VENV_DIR"
fi

if [ -f "$REQ_FILE" ]; then
  REQ_HASH="$(shasum -a 256 "$REQ_FILE" | awk '{print $1}')"
  INSTALLED_REQ_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"
  if [ ! -f "$STAMP_FILE" ] || [ "$REQ_FILE" -nt "$STAMP_FILE" ] || [ "$REQ_HASH" != "$INSTALLED_REQ_HASH" ]; then
    echo "正在安装/检查运行依赖，首次启动可能需要几分钟..."
    "$PIP_BIN" install --upgrade pip
    "$PIP_BIN" install -r "$REQ_FILE"
    date > "$STAMP_FILE"
    echo "$REQ_HASH" > "$REQ_HASH_FILE"
  fi
fi

echo "正在检查 Python 运行依赖..."
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import fastapi
import multipart
import pypdf
import fitz
import requests
import uvicorn
PY
then
  echo "检测到运行依赖不完整，正在重新安装依赖..."
  "$PIP_BIN" install --upgrade pip
  "$PIP_BIN" install -r "$REQ_FILE"
  date > "$STAMP_FILE"
  if [ -f "$REQ_FILE" ]; then
    shasum -a 256 "$REQ_FILE" | awk '{print $1}' > "$REQ_HASH_FILE"
  fi
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pypdf
PY
then
  echo
  echo "错误：PDF 解析依赖 pypdf 安装失败。课程 PDF 术语提取将不可用。"
  echo "请检查网络后重新双击 start_web.command，或在终端执行："
  echo "\"$PIP_BIN\" install -r \"$REQ_FILE\""
  echo
  read -r -p "按回车键关闭窗口..."
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import fitz
PY
then
  echo "提示：PyMuPDF 未安装成功。普通 PDF 仍可用，但部分特殊编码 PDF 的文字提取可能失败。"
  echo "如需增强 PDF 兼容性，可在项目目录执行：\"$PIP_BIN\" install PyMuPDF"
fi

export DRAUTOCUT_PDF_PYTHON="$PYTHON_BIN"

echo
echo "正在检查 FFmpeg / FFprobe..."
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  echo "  FFmpeg 和 FFprobe 已可用。"
else
  echo "  未检测到完整 FFmpeg 套件。视频探测、音频提取和渲染校验需要 ffmpeg/ffprobe。"
  if command -v brew >/dev/null 2>&1; then
    echo "  检测到 Homebrew，正在安装 ffmpeg..."
    brew install ffmpeg
    hash -r
  else
    echo "  未检测到 Homebrew，无法自动安装系统级 FFmpeg。"
    echo "  请先安装 Homebrew 后执行：brew install ffmpeg"
    echo "  Homebrew 安装地址：https://brew.sh"
  fi
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo
  echo "警告：ffprobe 仍不可用。"
  echo "如果继续启动 WebUI，开始处理视频时会报错：Missing required command: ffprobe"
  echo
  read -r -p "按回车键继续启动 WebUI，或关闭窗口后先安装 FFmpeg..."
elif ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "警告：ffmpeg 仍不可用。音频提取和部分流程会失败。"
  echo
  read -r -p "按回车键继续启动 WebUI，或关闭窗口后先安装 FFmpeg..."
fi

RESOLVE_APP="/Applications/DaVinci Resolve/DaVinci Resolve.app"
RESOLVE_MODULES="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/DaVinciResolveScript.py"
RESOLVE_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"

echo
echo "正在检查 DaVinci Resolve 环境..."
if [ ! -d "$RESOLVE_APP" ]; then
  echo "  未检测到 DaVinci Resolve。仍可启动 WebUI，但只能先生成 SRT，不能自动渲染成片。"
elif [ ! -f "$RESOLVE_MODULES" ]; then
  echo "  已检测到 DaVinci Resolve，但未找到官方脚本模块："
  echo "  $RESOLVE_MODULES"
  echo "  请确认安装的是包含脚本接口的 DaVinci Resolve，并检查 Developer/Scripting 目录。"
elif [ ! -f "$RESOLVE_LIB" ]; then
  echo "  已检测到 DaVinci Resolve 脚本模块，但未找到 Fusion 脚本库："
  echo "  $RESOLVE_LIB"
  echo "  自动渲染可能不可用。"
else
  echo "  Resolve 脚本接口路径已找到。"
  if ! pgrep -f "DaVinci Resolve" >/dev/null 2>&1; then
    echo "  提醒：需要自动导入/渲染时，请先打开 DaVinci Resolve，并启用 External scripting: Local。"
  fi
fi
echo "  说明：当前版本使用 Resolve 官方 Python 脚本接口，不需要安装 Resolve MCP。"
echo

(
  for _ in $(seq 1 80); do
    if curl -fsS "$STATUS_URL" >/dev/null 2>&1; then
      open "$URL" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 0.5
  done
  echo
  echo "服务已启动，但自动打开网页超时。请手动访问：$URL"
) &

echo "正在启动 Web 服务..."
echo "终端窗口保持打开期间，WebUI 会持续运行。"
echo "要关闭服务，在这个窗口按 Ctrl+C。"
echo

exec "$PYTHON_BIN" app/web_server.py --host "$HOST" --port "$PORT"
