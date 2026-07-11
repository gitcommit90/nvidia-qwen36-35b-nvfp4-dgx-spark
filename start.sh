#!/usr/bin/env bash
# Start nvidia/Qwen3.6-35B-A3B-NVFP4 on NVIDIA DGX Spark / GB10 via vLLM.
# Tested on ASUS Ascent GX10 (GB10, SM121) with eugr/spark-vllm.
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-35b-a3b-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME:-${WORK_DIR}/.cache/huggingface}"
LOG_FILE="${WORK_DIR}/.vllm.log"
PID_FILE="${WORK_DIR}/.vllm.pid"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_HOME}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} is already running."
  curl -sS "${READY_URL}" | head -c 400 || true
  echo
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing stopped container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

hf_cache_repo_dir() {
  echo "${HF_HOME}/hub/models--${1//\//--}"
}

model_is_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"
  [[ -d "${cache_dir}/snapshots" ]] || return 1
  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/"*.safetensors >/dev/null; then
      return 0
    fi
  done
  return 1
}

download_model() {
  local model_id="$1"
  echo "Downloading ${model_id} into ${HF_HOME}"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      hf download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      huggingface-cli download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi
  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)"
}

if ! model_is_cached "${MODEL_ID}"; then
  download_model "${MODEL_ID}"
else
  echo "Model cache found for ${MODEL_ID}"
fi

# JSON configs written to files to avoid shell/vLLM double-escaping bugs.
mkdir -p "${WORK_DIR}/.runtime"
cat > "${WORK_DIR}/.runtime/spec.json" <<'EOF'
{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}
EOF
cat > "${WORK_DIR}/.runtime/chatkw.json" <<'EOF'
{"preserve_thinking":true}
EOF
cat > "${WORK_DIR}/.runtime/gen.json" <<'EOF'
{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}
EOF

cat > "${WORK_DIR}/.runtime/serve.sh" <<EOF
#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
SPEC=\$(cat /runtime/spec.json)
CHAT=\$(cat /runtime/chatkw.json)
GEN=\$(cat /runtime/gen.json)
exec vllm serve ${MODEL_ID} \\
  --host ${HOST} \\
  --port ${PORT} \\
  --language-model-only \\
  --tensor-parallel-size 1 \\
  --trust-remote-code \\
  --kv-cache-dtype fp8 \\
  --attention-backend flashinfer \\
  --moe-backend marlin \\
  --gpu-memory-utilization ${GPU_MEM_UTIL} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
  --enable-chunked-prefill \\
  --async-scheduling \\
  --enable-prefix-caching \\
  --speculative-config "\$SPEC" \\
  --load-format fastsafetensors \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder \\
  --enable-auto-tool-choice \\
  --default-chat-template-kwargs "\$CHAT" \\
  --override-generation-config "\$GEN"
EOF
chmod +x "${WORK_DIR}/.runtime/serve.sh"

echo "Pulling image ${IMAGE} (if needed)"
docker pull "${IMAGE}" >/dev/null || true

echo "Starting ${CONTAINER_NAME} on port ${PORT} (max_num_seqs=${MAX_NUM_SEQS})"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --ipc=host \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "${HF_HOME}:/cache/huggingface" \
  -v "${WORK_DIR}/.runtime:/runtime:ro" \
  "${IMAGE}" \
  bash /runtime/serve.sh >/dev/null

docker inspect -f '{{.State.Pid}}' "${CONTAINER_NAME}" > "${PID_FILE}" || true

echo "Waiting for ${READY_URL}"
for i in $(seq 1 120); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    echo "Ready after ~$((i * 5))s"
    curl -sS "${READY_URL}"
    echo
    echo "Logs: docker logs -f ${CONTAINER_NAME}"
    echo "Stop:  ./stop.sh"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited during startup. Last logs:"
    docker logs --tail 80 "${CONTAINER_NAME}" || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for API. Last logs:"
docker logs --tail 80 "${CONTAINER_NAME}" || true
exit 1
