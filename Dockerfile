# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ARG COMFY_COMMIT=ee71d5c4993f29086b27fde1629a945ae48425bf
LABEL org.opencontainers.image.title="Maja — FLUX.2 klein 9B for RunPod"
LABEL org.opencontainers.image.description="ComfyUI, mandatory Maja LoRA, pinned models and native reference workflows"
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WORKSPACE=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/ComfyUI && cd /opt/ComfyUI \
    && git remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git fetch --depth 1 origin "$COMFY_COMMIT" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$COMFY_COMMIT"
COPY requirements.lock /opt/maja/requirements.lock
RUN python -m pip install --no-cache-dir -r /opt/maja/requirements.lock \
    && python -m pip check \
    && python -c "import torch; assert torch.__version__.startswith('2.10.0'); assert torch.version.cuda == '12.8'"
COPY app /opt/maja/app
COPY config /opt/maja/config
COPY tests /opt/maja/tests
RUN chmod +x /opt/maja/app/entrypoint.sh \
    && python -m unittest discover -s /opt/maja/tests -v
WORKDIR /opt/ComfyUI
# CPU-only import/startup test during image build; GPU inference is a separate deployment check.
RUN python main.py --cpu --disable-api-nodes --disable-all-custom-nodes --quick-test-for-ci
RUN python /opt/maja/app/validate_comfy.py
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=5s --start-period=60m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=4)" || exit 1
ENTRYPOINT ["/opt/maja/app/entrypoint.sh"]
CMD []
