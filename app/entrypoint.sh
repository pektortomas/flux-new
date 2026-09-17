#!/usr/bin/env bash
set -euo pipefail
umask 077
export WORKSPACE="${WORKSPACE:-/workspace}"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1
python /opt/maja/app/preflight.py
python /opt/maja/app/bootstrap.py
cd /opt/ComfyUI
exec python main.py \
  --listen 0.0.0.0 --port 8188 --disable-auto-launch \
  --models-directory "$WORKSPACE/models" \
  --input-directory "$WORKSPACE/input" \
  --output-directory "$WORKSPACE/output" \
  --temp-directory "$WORKSPACE/temp" \
  --user-directory "$WORKSPACE/user" \
  --use-pytorch-cross-attention --reserve-vram "${RESERVE_VRAM_GB:-2}" \
  --preview-method none --disable-api-nodes --disable-all-custom-nodes \
  --disable-comfy-compiler "$@"
