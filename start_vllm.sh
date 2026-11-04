
#!/bin/bash
# start_vllm.sh
conda activate muagent_vllm
chmod +x start_vllm.sh
MODEL_PATH=${1:-/home/robot/model/Qwen3-14B}
PORT=${2:-8000}
TP_SIZE=${3:-1}
SERVED_NAME=${4:-qwen3-14b}    # ← 新增第4个参数模型名

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
###启动vllm
###激活环境 conda activate muagent_vllm
#chmod +x start_vllm.sh
#
## 启动 Qwen3-14B（后台）
#nohup bash start_vllm.sh /home/robot/model/Qwen3-14B 8000 1 > vllm_qwen3.log 2>&1 &
#
## 观察启动日志（约 30–60 秒）
#tail -f vllm_qwen3.log
#
## 看到 "Uvicorn running on http://0.0.0.0:8000" 即表示启动完成


##验证服务
#curl http://localhost:8000/v1/models
##应看到
#{
#  "object": "list",
#  "data": [{"id": "qwen3-14b", "object": "model", ...}]
#}