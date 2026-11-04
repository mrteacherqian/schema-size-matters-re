
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

This project uses a **dual-environment architecture** to ensure runtime efficiency: the main environment handles code logic (LangGraph/data processing), while the vLLM environment is dedicated to serving the LLM API.
To facilitate reproduction, we provide both **Conda** and **Pip** installation methods. Please choose one based on your system.

### 1. Main Environment (muagents)
Used for running evaluation scripts, loading data, and building multi-agent graphs.

**Option A: Conda Installation (Full dependencies, for strict alignment)**
```bash
conda env create -f environment-main.yml
conda activate muagents
```
**Option B: Pip Installation (Minimal dependencies, cross-platform)**

```bash
conda create -n muagents python=3.10 -y
conda activate muagents
pip install -r requirements-main.txt
```
### 2. vLLM Environment (muagent_vllm)
Dedicated to loading LLMs and launching the OpenAI-compatible API service.

**Option A: Conda Installation (Full dependencies, suitable for Linux + CUDA 12.4)**
```bash
conda env create -f environment-vllm.yml
conda activate muagent_vllm
```
**Option B: Pip Installation (Minimal dependencies, Pip auto-adapts to your CUDA version)**
```bash
conda create -n muagent_vllm python=3.10 -y
conda activate muagent_vllm
pip install -r requirements-vllm.txt
```
## 📂 Data & Model Preparation
### 1. Data Preparation
Please download the datasets and place them in the data/ directory. The structure should look like this:
```text
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
```

### 2. Model Preparation
Please download the LLMs (e.g., Qwen3-14B, Qwen2.5-7B-Instruct) and the Sentence-BERT model (all-MiniLM-L6-v2) into the local model/ directory.
Note: For Qwen2.5-7B, the expected path is ./model/Qwen2.5-7B-Instruct.。
The default paths are configured in config.py and can be modified as needed.

### 3. Environment Variables
Create a .env file in the project root to configure your API routes (you can refer to .env.example):

```env
# ==========================================
# Model Selection
# ==========================================
# Options: "vllm", "local", or "deepseek"
# The main experiments in the paper use vLLM.
MODEL_PROVIDER=vllm

# ==========================================
# vLLM API Configuration (Recommended)
# ==========================================
# The Python scripts (e.g., run_semeval.py) will automatically override 
# VLLM_BASE_URL and VLLM_MODEL_NAME based on the `model_key` argument.
# 
# - For Qwen3-14B: `python run_semeval.py qwen3-14b`
#   It will use: http://localhost:8000/v1 (model: qwen3-14b)
# 
# - For Qwen2.5-7B: `python run_semeval.py qwen2.5-7b`
#   It will use: http://localhost:8001/v1 (model: qwen2.5-7b)

# Default fallback settings (usually for Qwen3-14B on port 8000)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL_NAME=qwen3-14b

# ==========================================
# Local HuggingFace Pipeline (Alternative)
# ==========================================
# If you set MODEL_PROVIDER=local, the code loads the model directly.
# Expected path for Qwen2.5-7B-Instruct:
LOCAL_MODEL_PATH=./model/Qwen2.5-7B-Instruct

# ==========================================
# Generation Parameters
# ==========================================
# Temperature=0.0 ensures deterministic outputs.
TEMPERATURE=0.0
# 50 tokens are sufficient since the model is instructed to output only a label.
MAX_TOKENS=50
```
## 🚀 Running Experiments
This project supports experiments on both Qwen3-14B and Qwen2.5-7B-Instruct. Depending on your hardware, you can start them simultaneously or separately.

### 1.Step 1: Start vLLM Servers (Terminal 1 & 2)
You need to assign an independent port for each model (default 8000 for Qwen3 and 8001 for Qwen2.5).
Ensure your terminal is in the project root directory (where start_vllm.sh is located).
Open a terminal, activate the vLLM environment, and grant execution permission:

```bash
# Activate vLLM environment
conda activate muagent_vllm

# Make script executable
chmod +x start_vllm.sh
```
**Approach A: Run both models simultaneously (in the background)**
(Note: Ensure your GPU memory is sufficient to hold both 14B and 7B models.)
```bash
# Start Qwen3-14B on port 8000 (background)
nohup bash start_vllm.sh /home/robot/model/Qwen3-14B 8000 1 qwen3-14b > vllm_qwen3.log 2>&1 &

# Start Qwen2.5-7B on port 8001 (background, note the relative path in ./model/)
nohup bash start_vllm.sh ./model/Qwen2.5-7B-Instruct 8001 1 qwen2.5-7b > vllm_qwen25.log 2>&1 &
```
**Approach B: Run separately (Recommended if you have limited GPU memory)**
Start Qwen3-14B first:
```bash
bash start_vllm.sh /home/robot/model/Qwen3-14B 8000 1 qwen3-14b
```
(Wait for Uvicorn running on http://0.0.0.0:8000, then run your experiments. After finishing, press Ctrl+C to stop, and then start Qwen2.5-7B):
```bash
bash start_vllm.sh ./model/Qwen2.5-7B-Instruct 8001 1 qwen2.5-7b
```
### 2.Step 2: Run Main Scripts (Terminal 3)
Open another terminal, activate the main environment, and run the experiments.

For Qwen3-14B (Port 8000):

```bash
# Activate main environment
conda activate muagents

# SemEval-2010 (19 classes)
python run_semeval.py qwen3-14b --provider vllm

# SciERC (7 classes)
python run_scierc.py qwen3-14b --provider vllm

# Wiki80 (80 classes)
python run_wiki80.py qwen3-14b --provider vllm
```
For Qwen2.5-7B (Port 8001):

```bash
# Activate main environment
conda activate muagents

# SemEval-2010 (19 classes)
python run_semeval.py qwen2.5-7b --provider vllm

# SciERC (7 classes)
python run_scierc.py qwen2.5-7b --provider vllm

# Wiki80 (80 classes)
python run_wiki80.py qwen2.5-7b --provider vllm
```
(Results will be automatically saved as JSON files in the outputs/ directory.)

### 3.Step 3: Sensitivity Analysis
Run the threshold sensitivity analysis (forcing Refiner activation at conf=60):
```bash
python run_sensitivity.py --models qwen3-14b qwen2.5-7b
```

### 4.Step 4: Extract Validator Confidence
Extract Validator confidence distribution from the execution logs (generates a JSON for plotting):
```bash
python extract_validator_conf.py
```
## 📁 Code Structure
agents.py: Defines Extractor, Validator, and Refiner agents; handles LangChain prompts and LLM initialization.

baselines.py: Implements baseline models, including PureLLM (Zero-shot) and FewShotPureLLM (with ICL demonstrations).

config.py: Global configuration, including API endpoints and file paths.

data_loader.py: Loads SemEval-2010, SciERC, and Wiki80 datasets, and handles entity tagging.

few_shot_examples.py: Implements 4 example selection strategies (Random, Similarity, Hybrid MMR, Class-Balanced).

graph.py: Core multi-agent workflow (LangGraph: Extractor → Validator → Refiner).

graph_extractor_only.py: Ablation study (Extractor → Evaluator).

graph_extractor_validator.py: Ablation study (Extractor → Validator → Evaluator, without Refiner).

state.py: Defines the shared AgentState and ExtractionResult data structures for LangGraph.

utils.py: Utility functions, including accuracy/Macro-F1 calculation and strict directional label parsing.

run_*.py: Execution pipelines for each dataset and the sensitivity analysis script.

start_vllm.sh: Shell script to launch the vLLM API service.

## 📝 Citation
If you find this code or our findings useful in your research, please consider citing our paper:

bibtex
@article{qian2026schema,
  title={Schema-Size Matters: An Empirical Study of Diversity-Aware Example Selection and the Limitations of Multi-Agent Verification for Few-Shot Relation Extraction},
  author={Qian, Xiangguo and Xu, Yong and Wei, Rikun},
  journal={[Journal Name]},
  year={2026}
}
## 📄 License
This project is licensed under the MIT License.
