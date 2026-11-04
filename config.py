# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
#
# class Config:
#     """全局配置"""
#
#     # ========== 模型选择 ==========
#     # deepseek: DeepSeek API (兼容OpenAI接口)
#     # local: 本地Qwen模型 (Transformers加载)
#     MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "deepseek")
#
#     # ========== DeepSeek API 配置 ==========
#     DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
#     DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
#     DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
#
#     # ========== 本地 Qwen 模型配置 ==========
#     # 模型放在 multi_agent_re/model/ 目录下
#     LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./model/Qwen2.5-7B-Instruct")
#
#     # ========== 通用配置 ==========
#     TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
#
#
#     # MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
#     # config.py - 修改 MAX_TOKENS
#     MAX_TOKENS = int(os.getenv("MAX_TOKENS", "50"))  # 从 512 改为 50
#     SEED = 42
#
#     # ========== 编码器配置 ==========
#     ENCODER_MODEL_PATH = "./model/all-MiniLM-L6-v2"  # 本地路径
#     # ENCODER_MODEL_PATH = "/home/robot/qian/multi_agent_re/model/all-MiniLM-L6-v2"
#
#
#     # ========== Self-RAG 配置 ==========
#     SELFRAG_MODEL = os.getenv("SELFRAG_MODEL", "selfrag/selfrag_llama2_7b")
#     SELFRAG_DTYPE = os.getenv("SELFRAG_DTYPE", "half")
#
#     # ========== 数据路径 ==========
#     DATA_DIR = "data"
#     SEMEVAL_TRAIN = "data/SemEval2010_task8_training/TRAIN_FILE.TXT"
#     SEMEVAL_TEST = "data/SemEval2010_task8_testing/TEST_FILE.TXT"
#     INDUSTRY_DATA = "data/industry_relations.jsonl"
#
#     # ========== 输出路径 ==========
#     OUTPUT_DIR = "outputs"
#     RESULTS_FILE = "outputs/results.json"
#
#     @classmethod
#     def get_llm_kwargs(cls):
#         """
#         获取 LLM 初始化参数
#         根据 MODEL_PROVIDER 返回对应参数字典
#         """
#         if cls.MODEL_PROVIDER == "deepseek":
#             # DeepSeek 使用 OpenAI 兼容接口
#             return {
#                 "model": cls.DEEPSEEK_MODEL,
#                 "temperature": cls.TEMPERATURE,
#                 "max_tokens": cls.MAX_TOKENS,
#                 "api_key": cls.DEEPSEEK_API_KEY,
#                 "base_url": cls.DEEPSEEK_API_BASE
#             }
#         elif cls.MODEL_PROVIDER == "local":
#             # 本地 Qwen 模型需要特殊处理
#             # 返回本地模型路径和加载参数
#             return {
#                 "model_path": cls.LOCAL_MODEL_PATH,
#                 "temperature": cls.TEMPERATURE,
#                 "max_tokens": cls.MAX_TOKENS,
#                 "device_map": "auto",  # 自动分配GPU/CPU
#                 "torch_dtype": "auto"   # 自动选择精度
#             }
#         else:
#             return {}
#
#     @classmethod
#     def get_provider_name(cls):
#         """获取当前提供者名称（用于日志）"""
#         if cls.MODEL_PROVIDER == "deepseek":
#             return f"DeepSeek ({cls.DEEPSEEK_MODEL})"
#         elif cls.MODEL_PROVIDER == "local":
#             return f"Local Qwen ({cls.LOCAL_MODEL_PATH})"
#         else:
#             return "Unknown"


# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """全局配置"""

    # ========== 模型选择 ==========
    # deepseek: DeepSeek API (兼容 OpenAI 接口)
    # local: 本地 HuggingFace 模型
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "local")

    # ========== DeepSeek API 配置 ==========
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ========== 本地模型配置 ==========
    # 主力模型：Qwen3-14B（绝对路径）
    LOCAL_MODEL_PATH = os.getenv(
        "LOCAL_MODEL_PATH",
        "/home/robot/model/Qwen3-14B"
    )
    # ========== vLLM 配置（新增）==========
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM 不校验，但 OpenAI 客户端需要非空
    VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-14b")

    # config.py - 增加 Qwen2.5-7B 的 vLLM 配置项
    # 预定义模型路径（便于切换）
    MODEL_PATHS = {
        "qwen3-14b":   "/home/robot/model/Qwen3-14B",
        "qwen2.5-7b":  "./model/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "./model/Qwen2.5-14B-Instruct",
        "qwen3-8b":    "./model/Qwen3-8B",
    }
    # 每个模型对应的 vLLM 端口和 serviced name
    VLLM_PORTS = {
        "qwen3-14b": ("http://localhost:8000/v1", "qwen3-14b"),
        "qwen2.5-7b": ("http://localhost:8001/v1", "qwen2.5-7b"),
    }
    # Qwen3 思考模式（默认关闭，保证模型间公平对比）
    ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

    # ========== 通用配置 ==========
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "50"))
    SEED = 42

    # ========== 编码器配置 ==========
    ENCODER_MODEL_PATH = "./model/all-MiniLM-L6-v2"

    # ========== Self-RAG 配置 ==========
    SELFRAG_MODEL = os.getenv("SELFRAG_MODEL", "./model/selfrag_llama2_7b")
    SELFRAG_DTYPE = os.getenv("SELFRAG_DTYPE", "half")

    # ========== 数据路径 ==========
    DATA_DIR = "data"

    # SemEval-2010 Task 8
    SEMEVAL_TRAIN = "data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    SEMEVAL_TEST  = "data/SemEval2010_task8_testing/TEST_FILE.TXT"

    # Wiki80 (OpenNRE 基准)
    WIKI80_TRAIN = "data/wiki80/wiki80_train.txt"
    WIKI80_VAL   = "data/wiki80/wiki80_val.txt"
    WIKI80_TEST  = "data/wiki80/wiki80_val.txt"

    # SciERC
    SCIERC_TRAIN = "data/scierc/scierc_train.json"
    SCIERC_DEV   = "data/scierc/scierc_dev.json"
    SCIERC_TEST  = "data/scierc/scierc_test.json"

    # 产业数据
    INDUSTRY_DATA = "data/industry_relations.jsonl"

    # ========== 输出路径 ==========
    OUTPUT_DIR = "outputs"
    RESULTS_FILE = "outputs/results.json"

    @classmethod
    def set_model(cls, model_key: str):
        """按 key 切换本地模型"""
        if model_key in cls.MODEL_PATHS:
            cls.LOCAL_MODEL_PATH = cls.MODEL_PATHS[model_key]
            print(f"[Config] 模型切换为: {model_key} -> {cls.LOCAL_MODEL_PATH}")
        else:
            raise ValueError(f"未知模型 key: {model_key}. 可选: {list(cls.MODEL_PATHS.keys())}")

    @classmethod
    def get_llm_kwargs(cls):
        if cls.MODEL_PROVIDER == "deepseek":
            return {
                "model": cls.DEEPSEEK_MODEL,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "api_key": cls.DEEPSEEK_API_KEY,
                "base_url": cls.DEEPSEEK_API_BASE,
            }
        elif cls.MODEL_PROVIDER == "local":
            return {
                "model_path": cls.LOCAL_MODEL_PATH,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "device_map": "auto",
                "torch_dtype": "float16",
                "enable_thinking": cls.ENABLE_THINKING,
            }
        elif cls.MODEL_PROVIDER == "vllm":
            return {
                "model": cls.VLLM_MODEL_NAME,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "api_key": cls.VLLM_API_KEY,
                "base_url": cls.VLLM_BASE_URL,
            }
        return {}