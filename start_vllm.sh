
#!/bin/bash
# start_vllm.sh
MODEL_PATH=${1:-/home/robot/model/Qwen3-14B}
PORT=${2:-8000}
TP_SIZE=${3:-1}
SERVED_NAME=${4:-qwen3-14b}    

echo "启动 vLLM 服务"
echo "  模型: ${MODEL_PATH}"
echo "  端口: ${PORT}"
echo "  served-name: ${SERVED_NAME}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size ${TP_SIZE} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --disable-log-requests
# ==================================================
# Start vLLM Server
# ==================================================
# 1. Activate the conda environment
# conda activate muagent_vllm

# 2. Grant execution permission (only needs to be done once)
# chmod +x start_vllm.sh

# 3. Start Qwen3-14B in the background
# nohup bash start_vllm.sh /home/robot/model/Qwen3-14B 8000 1 > vllm_qwen3.log 2>&1 &

# 4. Monitor the startup log (takes about 30–60 seconds)
# tail -f vllm_qwen3.log

# 5. The server is ready when you see "Uvicorn running on http://0.0.0.0:8000"

# ==================================================
# Verify the API service
# ==================================================
# curl http://localhost:8000/v1/models

# Expected output:
# {
#   "object": "list",
#   "data": [{"id": "qwen3-14b", "object": "model", ...}]
# }
