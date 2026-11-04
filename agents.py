# # agents.py - 完整修改版9.317.50
#
# import re
# import json
# import torch
# from typing import Dict, Any, Optional
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# from config import Config
# from utils import parse_relation_label
#
#
# class BaseAgent:
#     """智能体基类 - 支持 DeepSeek API 和本地 Qwen 模型"""
#
#     _cached_llm = None
#     _cached_provider = None
#
#     def __init__(self, model_provider: str = None, temperature: float = None):
#         self.model_provider = model_provider or Config.MODEL_PROVIDER
#         self.temperature = temperature or Config.TEMPERATURE
#         self._llm = None
#
#     @property
#     def llm(self):
#         if self._llm is not None:
#             return self._llm
#
#         if (BaseAgent._cached_llm is not None and
#                 BaseAgent._cached_provider == self.model_provider):
#             print(f"[LLM] 使用缓存的 {self.model_provider} 模型（避免重复加载）")
#             self._llm = BaseAgent._cached_llm
#             return self._llm
#
#         print(f"[LLM] 首次加载 {self.model_provider} 模型（将缓存）...")
#
#         if self.model_provider == "deepseek":
#             from langchain_openai import ChatOpenAI
#             kwargs = Config.get_llm_kwargs()
#             self._llm = ChatOpenAI(
#                 model=kwargs["model"],
#                 temperature=kwargs["temperature"],
#                 max_tokens=kwargs["max_tokens"],
#                 api_key=kwargs["api_key"],
#                 base_url=kwargs["base_url"]
#             )
#             print(f"[LLM] 已加载 DeepSeek: {kwargs['model']}")
#
#         elif self.model_provider == "local":
#             from transformers import AutoModelForCausalLM, AutoTokenizer
#             from langchain_huggingface import HuggingFacePipeline
#             from transformers import pipeline
#             import torch
#
#             kwargs = Config.get_llm_kwargs()
#             model_path = kwargs["model_path"]
#
#             print(f"[LLM] 正在加载本地 Qwen 模型: {model_path}")
#
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_path,
#                 trust_remote_code=True
#             )
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#
#             pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=50,
#                 temperature=0.1,
#                 do_sample=True,
#                 top_p=1.0,
#                 return_full_text=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id
#             )
#
#             self._llm = HuggingFacePipeline(pipeline=pipe)
#             print(f"[LLM] 本地 Qwen 模型加载完成")
#
#         else:
#             from langchain_openai import ChatOpenAI
#             self._llm = ChatOpenAI(
#                 model=Config.DEEPSEEK_MODEL,
#                 temperature=self.temperature,
#                 max_tokens=Config.MAX_TOKENS,
#                 api_key=Config.DEEPSEEK_API_KEY,
#                 base_url=Config.DEEPSEEK_API_BASE
#             )
#             print("[LLM] 使用默认 DeepSeek 模型")
#
#         BaseAgent._cached_llm = self._llm
#         BaseAgent._cached_provider = self.model_provider
#
#         return self._llm
#
#
# class ExtractorAgent(BaseAgent):
#     """抽取智能体"""
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.prompt = ChatPromptTemplate.from_template("""请判断以下句子中两个实体的关系。
#
# 句子: {sentence}
# 实体1: {e1}
# 实体2: {e2}
#
# 可选关系（只输出其中一个）:
# Cause-Effect, Component-Whole, Content-Container, Entity-Destination,
# Entity-Origin, Instrument-Agency, Member-Collection, Message-Topic,
# Product-Producer, Other
#
# 要求:
# - 关系有方向: {e1} → {e2}
# - 只输出关系名称，不要输出任何解释、标点或多余文字。
#
# 你的回答（只输出一个词）:""")
#         self.chain = self.prompt | self.llm | StrOutputParser()
#
#     def extract(self, sentence: str, e1: str, e2: str) -> Dict[str, Any]:
#         try:
#             response = self.chain.invoke({"sentence": sentence, "e1": e1, "e2": e2})
#             relation = parse_relation_label(response)
#             return {
#                 "sentence": sentence, "e1": e1, "e2": e2,
#                 "predicted_relation": relation,
#                 "raw_response": response,
#                 "confidence": 0.7,
#                 "error": ""
#             }
#         except Exception as e:
#             return {
#                 "sentence": sentence, "e1": e1, "e2": e2,
#                 "predicted_relation": "Other",
#                 "raw_response": "",
#                 "confidence": 0.0,
#                 "error": str(e)
#             }
#
#
# # ========== 优化1：ValidatorAgent 提示词增加 suggested_relation ==========
# class ValidatorAgent(BaseAgent):
#     """验证智能体 - 优化版"""
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         # ===== 优化：增加 suggested_relation 输出 =====
#         self.prompt = ChatPromptTemplate.from_template("""你是一个关系抽取结果验证专家。
#
# 句子: {sentence}
# 实体1: {e1}
# 实体2: {e2}
# 候选关系: {candidate_relation}
#
# **任务**：验证候选关系是否正确。
#
# **验证维度**：
# 1. 语义合理性：候选关系是否符合句子含义？
# 2. 实体兼容性：实体类型是否与该关系兼容？
# 3. 方向正确性：{e1} → {e2} 的方向是否正确？
#
# **判断标准**：
# - 如果候选关系在语义上合理，判定为通过（passed: true）
# - 如果候选关系明显错误，判定为不通过（passed: false）
# - 如果你认为有其他更合适的关系，请在 suggested_relation 中给出
#
# **输出JSON格式**：
# {{"passed": true/false, "confidence": 0-100, "reason": "...", "suggested_relation": "..."}}
#
# 可选关系: Cause-Effect, Component-Whole, Content-Container, Entity-Destination,
# Entity-Origin, Instrument-Agency, Member-Collection, Message-Topic, Product-Producer, Other
#
# 只输出JSON，不要其他内容。""")
#         self.chain = self.prompt | self.llm | StrOutputParser()
#
#     def validate(self, sentence: str, e1: str, e2: str, candidate: str) -> Dict[str, Any]:
#         try:
#             response = self.chain.invoke({
#                 "sentence": sentence, "e1": e1, "e2": e2, "candidate_relation": candidate
#             })
#
#             # 提取JSON
#             json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
#                 return {
#                     "passed": result.get("passed", False),
#                     "confidence": result.get("confidence", 50),
#                     "reason": result.get("reason", ""),
#                     "suggested_relation": result.get("suggested_relation", candidate),
#                     "raw_response": response
#                 }
#
#             # 降级解析
#             passed = "通过" in response or "true" in response.lower()
#             conf_match = re.search(r'(\d+)', response)
#
#             # 尝试提取 suggested_relation
#             suggested = candidate
#             relations = ['Cause-Effect', 'Component-Whole', 'Content-Container',
#                          'Entity-Destination', 'Entity-Origin', 'Instrument-Agency',
#                          'Member-Collection', 'Message-Topic', 'Product-Producer']
#             for rel in relations:
#                 if rel in response:
#                     suggested = rel
#                     break
#
#             return {
#                 "passed": passed,
#                 "confidence": int(conf_match.group(1)) if conf_match else 50,
#                 "reason": response[:200],
#                 "suggested_relation": suggested,
#                 "raw_response": response
#             }
#         except Exception as e:
#             return {
#                 "passed": False,
#                 "confidence": 0,
#                 "reason": str(e),
#                 "suggested_relation": "Other",
#                 "raw_response": ""
#             }
#
#
# # ========== 优化2：RefinerAgent 提示词增强 ==========
# class RefinerAgent(BaseAgent):
#     """精炼智能体 - 优化版"""
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         # ===== 优化：更详细的精炼规则 =====
#         self.prompt = ChatPromptTemplate.from_template("""综合以下信息，给出最终的关系判定。
#
# 句子: {sentence}
# 实体1: {e1}
# 实体2: {e2}
#
# 抽取器的初步判定: {extractor_result}
# 验证器的反馈: {validator_feedback}
# 验证器的置信度: {validator_confidence}
#
# **精炼规则**：
# 1. 如果验证器置信度 >= 70，优先采纳验证器的判定
# 2. 如果验证器置信度在 50-70 之间，综合判断
# 3. 如果验证器置信度 < 50，保留抽取器结果
# 4. 如果验证器提供了 suggested_relation，考虑采纳
#
# **必须输出一个有效的可选关系名**。
#
# 只输出关系名称，不要输出解释。
#
# 可选关系: Cause-Effect, Component-Whole, Content-Container, Entity-Destination,
# Entity-Origin, Instrument-Agency, Member-Collection, Message-Topic, Product-Producer, Other
#
# 你的回答:""")
#         self.chain = self.prompt | self.llm | StrOutputParser()
#
#     def refine(self, sentence: str, e1: str, e2: str,
#                extractor_result: str, validator_feedback: str,
#                validator_confidence: int) -> Dict[str, Any]:
#         try:
#             response = self.chain.invoke({
#                 "sentence": sentence, "e1": e1, "e2": e2,
#                 "extractor_result": extractor_result,
#                 "validator_feedback": validator_feedback,
#                 "validator_confidence": validator_confidence
#             })
#             return {"final_relation": parse_relation_label(response), "error": ""}
#         except Exception as e:
#             return {"final_relation": extractor_result, "error": str(e)}

# agents.py
import re
import json
import torch
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config
from utils import parse_relation_label, SEMEVAL_19

# agents.py - BaseAgent.llm 属性替换

# class BaseAgent:
#     """智能体基类 - 支持 DeepSeek / 本地 HF / vLLM 三种后端"""
#
#     _cached_llm = None
#     _cached_provider = None
#     _cached_model_path = None
#
#     def __init__(self, model_provider: str = None, temperature: float = None):
#         self.model_provider = model_provider or Config.MODEL_PROVIDER
#         self.temperature = temperature if temperature is not None else Config.TEMPERATURE
#         self._llm = None
#
#     @property
#     def llm(self):
#         if self._llm is not None:
#             return self._llm
#
#         # 缓存键：provider + 具体模型标识
#         if self.model_provider == "vllm":
#             cache_key = ("vllm", Config.VLLM_BASE_URL, Config.VLLM_MODEL_NAME)
#         elif self.model_provider == "local":
#             cache_key = ("local", Config.LOCAL_MODEL_PATH)
#         elif self.model_provider == "deepseek":
#             cache_key = ("deepseek", Config.DEEPSEEK_MODEL)
#         else:
#             cache_key = (self.model_provider,)
#
#         if (BaseAgent._cached_llm is not None
#                 and BaseAgent._cached_model_path == cache_key):
#             print(f"[LLM] 复用缓存: {cache_key}")
#             self._llm = BaseAgent._cached_llm
#             return self._llm
#
#         print(f"[LLM] 首次加载: provider={self.model_provider}, key={cache_key}")
#
#         # ========== 分支1: DeepSeek API ==========
#         if self.model_provider == "deepseek":
#             from langchain_openai import ChatOpenAI
#             kwargs = Config.get_llm_kwargs()
#             self._llm = ChatOpenAI(
#                 model=kwargs["model"],
#                 temperature=kwargs["temperature"],
#                 max_tokens=kwargs["max_tokens"],
#                 api_key=kwargs["api_key"],
#                 base_url=kwargs["base_url"],
#             )
#             print(f"[LLM] DeepSeek 加载完成: {kwargs['model']}")
#
#         # ========== 分支2: 本地 HuggingFace pipeline（保持原样）==========
#         elif self.model_provider == "local":
#             from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#             from langchain_huggingface import HuggingFacePipeline
#             import torch
#
#             kwargs = Config.get_llm_kwargs()
#             model_path = kwargs["model_path"]
#             print(f"[LLM] 加载本地模型: {model_path}")
#
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_path, trust_remote_code=True, padding_side="left"
#             )
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True,
#             )
#
#             pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=kwargs.get("max_tokens", 50),
#                 temperature=kwargs.get("temperature", 0.1),
#                 do_sample=True,
#                 top_p=1.0,
#                 return_full_text=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
#             self._llm = HuggingFacePipeline(pipeline=pipe)
#             print(f"[LLM] 本地模型加载完成")
#
#         # ========== 分支3: vLLM HTTP 服务（新增）==========
#         # elif self.model_provider == "vllm":
#         #     from langchain_openai import OpenAI
#         #     kwargs = Config.get_llm_kwargs()
#         #
#         #     # 用 Completion 接口（/v1/completions），与 HuggingFacePipeline 行为一致
#         #     # 避免 chat template 触发 Qwen3 的 thinking 模式
#         #     self._llm = OpenAI(
#         #         model=kwargs["model"],
#         #         temperature=kwargs["temperature"],
#         #         max_tokens=kwargs["max_tokens"],
#         #         api_key=kwargs["api_key"],
#         #         base_url=kwargs["base_url"],
#         #         # 关闭重试，让错误快速暴露
#         #         max_retries=2,
#         #         request_timeout=60,
#         #     )
#         #     print(f"[LLM] vLLM 加载完成: {kwargs['base_url']} / {kwargs['model']}")
#         elif self.model_provider == "vllm":
#             from langchain_openai import OpenAI
#             kwargs = Config.get_llm_kwargs()
#             self._llm = OpenAI(
#                 model=kwargs["model"],
#                 temperature=0.0,
#                 max_tokens=kwargs["max_tokens"],
#                 api_key=kwargs["api_key"],
#                 base_url=kwargs["base_url"],
#                 max_retries=2,
#                 request_timeout=60,
#             )
#             print(f"[LLM] vLLM 加载完成: {kwargs['base_url']} / {kwargs['model']}")
#         else:
#             raise ValueError(f"未知 provider: {self.model_provider}")
#
#         BaseAgent._cached_llm = self._llm
#         BaseAgent._cached_provider = self.model_provider
#         BaseAgent._cached_model_path = cache_key
#         return self._llm
# class BaseAgent:
#     """智能体基类 - 支持 DeepSeek API 和本地 HuggingFace 模型"""
#
#     _cached_llm = None
#     _cached_provider = None
#     _cached_model_path = None
#
#     def __init__(self, model_provider: str = None, temperature: float = None):
#         self.model_provider = model_provider or Config.MODEL_PROVIDER
#         self.temperature = temperature if temperature is not None else Config.TEMPERATURE
#         self._llm = None
#
#     @property
#     def llm(self):
#         if self._llm is not None:
#             return self._llm
#
#         cache_key = (self.model_provider, Config.LOCAL_MODEL_PATH)
#         if (BaseAgent._cached_llm is not None
#                 and BaseAgent._cached_provider == self.model_provider
#                 and BaseAgent._cached_model_path == Config.LOCAL_MODEL_PATH):
#             print(f"[LLM] 复用缓存模型: {self.model_provider} ({Config.LOCAL_MODEL_PATH})")
#             self._llm = BaseAgent._cached_llm
#             return self._llm
#
#         print(f"[LLM] 首次加载: provider={self.model_provider}, path={Config.LOCAL_MODEL_PATH}")
#
#         if self.model_provider == "deepseek":
#             from langchain_openai import ChatOpenAI
#             kwargs = Config.get_llm_kwargs()
#             self._llm = ChatOpenAI(
#                 model=kwargs["model"],
#                 temperature=kwargs["temperature"],
#                 max_tokens=kwargs["max_tokens"],
#                 api_key=kwargs["api_key"],
#                 base_url=kwargs["base_url"],
#             )
#             print(f"[LLM] DeepSeek 加载完成: {kwargs['model']}")
#                 elif self.model_provider == "local":
#             from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#             from langchain_huggingface import HuggingFacePipeline
#
#             kwargs = Config.get_llm_kwargs()
#             model_path = kwargs["model_path"]
#             print(f"[LLM] 加载本地模型: {model_path}")
#
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_path, trust_remote_code=True, padding_side="left"
#             )
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True,
#             )
#
#             # 关键：do_sample=False（贪心解码），避免 Qwen3 自问自答；
#             # max_new_tokens 用 Config.MAX_TOKENS（50），足够输出一个关系名
#             pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=kwargs.get("max_tokens", 50),
#                 do_sample=False,
#                 return_full_text=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
#
#             self._llm = HuggingFacePipeline(pipeline=pipe)
#             print(f"[LLM] 本地模型加载完成")
#         elif self.model_provider == "local":
#             from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#             from langchain_huggingface import HuggingFacePipeline
#
#             kwargs = Config.get_llm_kwargs()
#             model_path = kwargs["model_path"]
#
#             print(f"[LLM] 加载本地模型: {model_path}")
#
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_path, trust_remote_code=True, padding_side="left"
#             )
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True,
#             )
#
#             pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_new_tokens=kwargs.get("max_tokens", 50),
#                 temperature=kwargs.get("temperature", 0.1),
#                 do_sample=True,
#                 top_p=1.0,
#                 return_full_text=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
#
#             self._llm = HuggingFacePipeline(pipeline=pipe)
#             print(f"[LLM] 本地模型加载完成")
#         else:
#             raise ValueError(f"未知 provider: {self.model_provider}")
#
#         BaseAgent._cached_llm = self._llm
#         BaseAgent._cached_provider = self.model_provider
#         BaseAgent._cached_model_path = Config.LOCAL_MODEL_PATH
#         return self._llm

# agents.py
import re
import json
import torch
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config
from utils import parse_relation_label, SEMEVAL_19


# def make_prompt(template: str, provider: str):
#     """
#     根据 provider 选择 prompt 类型：
#       - local:    PromptTemplate（HuggingFacePipeline 是 BaseLLM，接收字符串，
#                   避免走 chat template 导致"自问自答"）
#       - vllm:     ChatPromptTemplate（ChatOpenAI 走 chat 接口）
#       - deepseek: ChatPromptTemplate（ChatOpenAI 走 chat 接口）
#     """
#     if provider == "local":
#         return PromptTemplate.from_template(template)
#     return ChatPromptTemplate.from_template(template)
def make_prompt(template: str, provider: str):
    """
    - deepseek:  ChatPromptTemplate（走 chat 接口）
    - local/vllm: PromptTemplate（走纯文本补全，配合 /no_think 抑制 Qwen3 思考）
    """
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    if provider == "deepseek":
        return ChatPromptTemplate.from_template(template)
    return PromptTemplate.from_template(template)

class BaseAgent:
    """智能体基类 - 支持 DeepSeek / 本地 HF / vLLM 三种后端"""

    _cached_llm = None
    _cached_provider = None
    _cached_model_path = None

    def __init__(self, model_provider: str = None, temperature: float = None):
        self.model_provider = model_provider or Config.MODEL_PROVIDER
        self.temperature = temperature if temperature is not None else Config.TEMPERATURE
        self._llm = None

    @property
    def llm(self):
        if self._llm is not None:
            return self._llm

        # 缓存键：provider + 具体模型标识
        if self.model_provider == "vllm":
            cache_key = ("vllm", Config.VLLM_BASE_URL, Config.VLLM_MODEL_NAME)
        elif self.model_provider == "local":
            cache_key = ("local", Config.LOCAL_MODEL_PATH)
        elif self.model_provider == "deepseek":
            cache_key = ("deepseek", Config.DEEPSEEK_MODEL)
        else:
            cache_key = (self.model_provider,)

        if (BaseAgent._cached_llm is not None
                and BaseAgent._cached_model_path == cache_key):
            print(f"[LLM] 复用缓存: {cache_key}")
            self._llm = BaseAgent._cached_llm
            return self._llm

        print(f"[LLM] 首次加载: provider={self.model_provider}, key={cache_key}")

        # ========== 分支1: DeepSeek API ==========
        if self.model_provider == "deepseek":
            from langchain_openai import ChatOpenAI
            kwargs = Config.get_llm_kwargs()
            self._llm = ChatOpenAI(
                model=kwargs["model"],
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
            )
            print(f"[LLM] DeepSeek 加载完成: {kwargs['model']}")

        # ========== 分支2: 本地 HuggingFace pipeline ==========
        elif self.model_provider == "local":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_huggingface import HuggingFacePipeline

            kwargs = Config.get_llm_kwargs()
            model_path = kwargs["model_path"]
            print(f"[LLM] 加载本地模型: {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            # 关键：do_sample=False（贪心解码），避免 Qwen3 自问自答；
            # max_new_tokens 用 Config.MAX_TOKENS（50），足够输出一个关系名
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=kwargs.get("max_tokens", 50),
                do_sample=False,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            self._llm = HuggingFacePipeline(pipeline=pipe)
            print(f"[LLM] 本地模型加载完成")

        # ========== 分支3: vLLM HTTP 服务 ==========
        elif self.model_provider == "vllm":
            from langchain_openai import ChatOpenAI
            kwargs = Config.get_llm_kwargs()

            # 用 ChatOpenAI（走 /v1/chat/completions），
            # 配合 vLLM 启动参数 --default-chat-template-kwargs '{"enable_thinking": false}'
            # 从服务端彻底关闭 Qwen3 思考模式
            self._llm = ChatOpenAI(
                model=kwargs["model"],
                temperature=0.0,            # 强制确定性
                max_tokens=kwargs["max_tokens"],
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                max_retries=2,
                request_timeout=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            print(f"[LLM] vLLM 加载完成: {kwargs['base_url']} / {kwargs['model']}")

        else:
            raise ValueError(f"未知 provider: {self.model_provider}")

        BaseAgent._cached_llm = self._llm
        BaseAgent._cached_provider = self.model_provider
        BaseAgent._cached_model_path = cache_key
        return self._llm


class ExtractorAgent(BaseAgent):
    """抽取智能体"""

    def __init__(self, relation_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.relation_names = relation_names or SEMEVAL_19
        self.is_directional = any('(' in r for r in self.relation_names)

        rel_list = "\n".join(f"  {i}. {r}" for i, r in enumerate(self.relation_names, 1))
        direction_hint = ""
        if self.is_directional:
            direction_hint = (
                "\n【方向要求】关系具有方向性：\n"
                "  - e1 是原因、e2 是结果 → Cause-Effect(e1,e2)\n"
                "  - e2 是原因、e1 是结果 → Cause-Effect(e2,e1)\n"
                "  必须带 (e1,e2) 或 (e2,e1) 后缀。\n"
            )

        template = (
                "判断以下句子中两个实体的关系。\n\n"
                "句子: {sentence}\n"
                "实体1: {e1}\n"
                "实体2: {e2}\n\n"
                "可选关系（只输出一个）:\n"
                + "\n".join(f"- {r}" for r in self.relation_names) + "\n\n"
                + direction_hint
                + "\n只输出关系名。\n\n"
                  "你的回答:"
        )
        self.prompt = make_prompt(template, self.model_provider)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def extract(self, sentence: str, e1: str, e2: str) -> Dict[str, Any]:
        try:
            response = self.chain.invoke({"sentence": sentence, "e1": e1, "e2": e2})
            relation = parse_relation_label(response, valid_relations=self.relation_names)
            return {
                "sentence": sentence, "e1": e1, "e2": e2,
                "predicted_relation": relation,
                "raw_response": response,
                "confidence": 0.7,
                "error": "",
            }
        except Exception as e:
            return {
                "sentence": sentence, "e1": e1, "e2": e2,
                "predicted_relation": self.relation_names[-1],
                "raw_response": "", "confidence": 0.0, "error": str(e),
            }


class ValidatorAgent(BaseAgent):
    """验证智能体"""

    def __init__(self, relation_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.relation_names = relation_names or SEMEVAL_19

        template = (
            "你是一个关系抽取结果验证专家。\n\n"
            "句子: {sentence}\n"
            "实体1: {e1}\n"
            "实体2: {e2}\n"
            "候选关系: {candidate_relation}\n\n"
            "**任务**：验证候选关系是否正确。\n\n"
            "**验证维度**：\n"
            "1. 语义合理性：候选关系是否符合句子含义？\n"
            "2. 实体兼容性：实体类型是否与该关系兼容？\n"
            "3. 方向正确性：{e1} → {e2} 的方向是否正确？\n\n"
            "**判断标准**：\n"
            "- 候选关系语义合理 → passed: true\n"
            "- 候选关系明显错误 → passed: false\n"
            "- 有更合适的关系 → 填入 suggested_relation\n\n"
            "**输出JSON格式**（只输出JSON）：\n"
            '{{"passed": true/false, "confidence": 0-100, '
            '"reason": "...", "suggested_relation": "..."}}\n\n'
            "可选关系: " + ", ".join(self.relation_names) + "\n\n"
            "/no_think"
        )
        self.prompt = make_prompt(template, self.model_provider)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def validate(self, sentence: str, e1: str, e2: str, candidate: str) -> Dict[str, Any]:
        try:
            response = self.chain.invoke({
                "sentence": sentence, "e1": e1, "e2": e2, "candidate_relation": candidate
            })
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return {
                        "passed": bool(result.get("passed", False)),
                        "confidence": int(result.get("confidence", 50)),
                        "reason": str(result.get("reason", "")),
                        "suggested_relation": str(result.get("suggested_relation", candidate)),
                        "raw_response": response,
                    }
                except (json.JSONDecodeError, ValueError):
                    pass


            # 降级解析
            passed = "true" in response.lower() or "通过" in response
            conf_match = re.search(r'(\d+)', response)
            suggested = candidate
            for rel in sorted(self.relation_names, key=len, reverse=True):
                if rel.lower() in response.lower():
                    suggested = rel
                    break
            return {
                "passed": passed,
                "confidence": int(conf_match.group(1)) if conf_match else 50,
                "reason": response[:200],
                "suggested_relation": suggested,
                "raw_response": response,
            }
        except Exception as e:
            return {
                "passed": False, "confidence": 0, "reason": str(e),
                "suggested_relation": self.relation_names[-1],
                "raw_response": "",
            }


class RefinerAgent(BaseAgent):
    """精炼智能体"""

    def __init__(self, relation_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.relation_names = relation_names or SEMEVAL_19

        template = (
            "综合以下信息，给出最终的关系判定。\n\n"
            "句子: {sentence}\n"
            "实体1: {e1}\n"
            "实体2: {e2}\n\n"
            "抽取器的初步判定: {extractor_result}\n"
            "验证器的反馈: {validator_feedback}\n"
            "验证器的置信度: {validator_confidence}\n\n"
            "验证器的建议关系: {validator_suggested}\n\n"     # ← 新增
            "**精炼规则**：\n"
            "1. 验证器置信度 >= 70，优先采纳验证器判定\n"
            "2. 置信度 50-70，综合判断\n"
            "3. 置信度 < 50，保留抽取器结果\n"
            "4. 有 suggested_relation 则考虑采纳\n\n"
            "**必须输出一个有效的可选关系名**。\n"
            "只输出关系名称，不要输出解释。\n\n"
            "可选关系: " + ", ".join(self.relation_names) + "\n\n"
            "你的回答: /no_think"
        )
        self.prompt = make_prompt(template, self.model_provider)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def refine(self, sentence, e1, e2,
               extractor_result, validator_feedback,
               validator_confidence,validator_suggested="") -> Dict[str, Any]:
        try:
            response = self.chain.invoke({
                "sentence": sentence, "e1": e1, "e2": e2,
                "extractor_result": extractor_result,
                "validator_feedback": validator_feedback,
                "validator_confidence": validator_confidence,
                "validator_suggested": validator_suggested or extractor_result,
            })
            return {
                "final_relation": parse_relation_label(response, self.relation_names),
                "error": "",
            }
        except Exception as e:
            return {"final_relation": extractor_result, "error": str(e)}

