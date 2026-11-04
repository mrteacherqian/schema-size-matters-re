
# Schema-Size Matters (RE)
This repository contains the official implementation for the paper:
**"Schema-Size Matters: An Empirical Study of Diversity-Aware Example Selection and the Limitations of Multi-Agent Verification for Few-Shot Relation Extraction"**

## 📌 Abstract
Few-shot relation extraction (RE) with large language models (LLMs) has emerged as a promising paradigm for low-resource information extraction. This study systematically evaluates two orthogonal levers: diversity-aware example selection (Random, Similarity, MMR, Class-Balanced) and multi-agent post-hoc verification (Extractor–Validator–Refiner). Across three benchmarks with distinct schema sizes—SciERC (7 classes), SemEval-2010 (19 classes), and Wiki80 (80 classes)—we demonstrate that example selection is a second-order factor, while schema size and model capacity dominate. We further reveal that under conservative thresholds, multi-agent verification is structurally near-inert, and under aggressive thresholds, it becomes unreliable.

## 🌟 Key Features
- **ICL Example Selection**: Implementation of 4 strategies: Random, Semantic Similarity, Hybrid (MMR), and Class-Balanced.
- **Multi-Agent Pipeline**: A LangGraph-based **Extractor–Validator–Refiner** architecture.
- **Comprehensive Ablation**: Detailed scripts for Extractor-only and Extractor+Validator baselines.
- **Datasets Supported**: Ready-to-use loaders for SemEval-2010 Task 8, SciERC, and Wiki80.
- **Models Supported**: Supports vLLM (Qwen3-14B, Qwen2.5-7B) and DeepSeek APIs.

---

## 🛠️ Environment Setup

本项目采用**双环境架构**以保障运行效率：主环境负责运行代码逻辑（LangGraph/数据处理），vLLM 环境专门负责提供大模型 API 服务。
为了最大程度方便复现，本项目提供了 **Conda** 和 **Pip** 两种安装方式，请根据你的系统情况二选一。

### 1. Main Environment (主环境)
用于运行评估脚本、加载数据、构建多智能体图。

**选项 A：使用 Conda 安装（全量依赖，适合严格对齐底层环境）**
```bash
conda env create -f environment-main.yml
conda activate muagents
```

**选项 B：使用 Pip 安装（极简依赖，适合跨平台/其他 CUDA 版本）**
```bash
conda create -n muagents python=3.10 -y
conda activate muagents
pip install -r requirements-main.txt
```
### 2. vLLM Environment (模型服务环境)
专门用于加载大语言模型并启动 OpenAI 兼容的 API 服务。

选项 A：使用 Conda 安装（全量依赖，适合 Linux + CUDA 12.4 环境）
```bash
conda env create -f environment-vllm.yml
conda activate muagent_vllm
```
选项 B：使用 Pip 安装（极简依赖，Pip 会自动适配系统的 CUDA 版本）
```bash
conda create -n muagent_vllm python=3.10 -y
conda activate muagent_vllm
pip install -r requirements-vllm.txt
```
## 📂 Data & Model Preparation
1. Data Preparation
请下载对应的数据集并放在 data/ 目录下。目录结构应如下：

text
data/
├── SemEval2010_task8_training/TRAIN_FILE.TXT
├── SemEval2010_task8_testing/TEST_FILE.TXT
├── scierc/
│   ├── scierc_train.json
│   └── scierc_test.json
└── wiki80/
    ├── wiki80_train.txt
    └── wiki80_val.txt
(Note: The data/ directory is ignored by git to avoid large file uploads.)

2. Model Preparation
请将本地模型（如 Qwen3-14B）以及用于示例检索的 Sentence-BERT 模型（all-MiniLM-L6-v2）下载到本地 model/ 目录中。
默认路径配置在 config.py 中，你可以根据需要修改。

3. Environment Variables
在项目根目录创建一个 .env 文件，配置你的 API 路由（可参考 .env.example）：

env
MODEL_PROVIDER=vllm

# vLLM settings
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL_NAME=qwen3-14b

# DeepSeek API (Optional)
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
🚀 Running Experiments
Step 1: Start vLLM Server (Terminal 1)
打开一个终端，激活 vLLM 环境，启动模型服务（以 Qwen3-14B 为例，默认端口 8000）：

```bash
# 激活 vLLM 环境
conda activate muagent_vllm

# 赋予脚本执行权限
chmod +x start_vllm.sh

# 启动模型
bash start_vllm.sh /home/robot/model/Qwen3-14B 8000 1 qwen3-14b
(等待终端输出 Uvicorn running on http://0.0.0.0:8000 即表示启动完成)
```

Step 2: Run Main Scripts (Terminal 2)
打开另一个终端，激活主环境，开始跑实验：

```bash
# 激活主环境
conda activate muagents

# SemEval-2010 (19 classes)
python run_semeval.py qwen3-14b --provider vllm

# SciERC (7 classes)
python run_scierc.py qwen3-14b --provider vllm

# Wiki80 (80 classes)
python run_wiki80.py qwen3-14b --provider vllm
```
结果将自动保存在 outputs/ 目录下生成 JSON 文件。

Step 3: Sensitivity Analysis
运行阈值敏感性分析实验（强制在 conf=60 时触发 Refiner）：

```bash
python run_sensitivity.py --models qwen3-14b qwen2.5-7b
```
Step 4: Extract Validator Confidence
从执行日志中提取 Validator 的置信度分布（生成 JSON 供绘图使用）：

```bash
python extract_validator_conf.py
```
## 📁 Code Structure
agents.py: 定义 Extractor, Validator, Refiner 智能体，处理 LangChain 提示词和 LLM 初始化。

baselines.py: 实现基线模型，包括 PureLLM (Zero-shot) 和 FewShotPureLLM (带 ICL 示例)。

config.py: 全局配置，包括 API 端点、文件路径等。

data_loader.py: 负责加载 SemEval-2010, SciERC, Wiki80 数据集，处理实体标记。

few_shot_examples.py: 实现 4 种示例选择策略 (Random, Similarity, Hybrid MMR, Class-Balanced)。

graph.py: 核心多智能体工作流 (LangGraph: Extractor → Validator → Refiner)。

graph_extractor_only.py: 消融实验 (Extractor → Evaluator)。

graph_extractor_validator.py: 消融实验 (Extractor → Validator → Evaluator，无 Refiner)。

state.py: 定义 LangGraph 的共享状态 AgentState 及 ExtractionResult 数据结构。

utils.py: 辅助函数，包括准确率/Macro-F1计算、严格方向匹配标签解析等。

run_*.py: 各数据集的具体执行管线及敏感度分析脚本。

start_vllm.sh: 用于一键启动 vLLM API 服务的脚本。

## 📝 Citation
If you find this code or our findings useful in your research, please consider citing our paper:

bibtex
@article{qian2026schema,
  title={Schema-Size Matters: An Empirical Study of Diversity-Aware Example Selection and the Limitations of Multi-Agent Verification for Few-Shot Relation Extraction},
  author={Qian, Xiangguo and Xu, Yong and Fu, Kui},
  journal={[Journal Name]},
  year={2026}
}
## 📄 License
This project is licensed under the MIT License.
