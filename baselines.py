

# baselines.py
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config
from agents import BaseAgent, make_prompt
from utils import parse_relation_label, compute_accuracy, SEMEVAL_19


class PureLLM(BaseAgent):
    """Pure LLM 基线（无示例）"""

    def __init__(self, relation_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.relation_names = relation_names or SEMEVAL_19

        rel_list = "\n".join(f"- {r}" for r in self.relation_names)
        direction_hint = ""
        if any('(' in r for r in self.relation_names):
            direction_hint = "\n【注意】必须带 (e1,e2) 或 (e2,e1) 方向后缀。\n"

        template = (
            "判断以下句子中两个实体的关系。\n\n"
            "句子: {sentence}\n"
            "实体1: {e1}\n"
            "实体2: {e2}\n\n"
            "可选关系（只输出其中一个）:\n"
            + rel_list + "\n"
            + direction_hint
            + "\n【强制要求】只输出关系名称，不要推理，不要解释，不要续写。\n\n"
            "你的回答: /no_think"
        )
        self.prompt = make_prompt(template, self.model_provider)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def predict(self, sentence: str, e1: str, e2: str) -> str:
        try:
            raw = self.chain.invoke({"sentence": sentence, "e1": e1, "e2": e2})
            return parse_relation_label(raw, valid_relations=self.relation_names)
        except Exception as e:
            print(f"PureLLM预测错误: {e}")
            return self.relation_names[-1]

    def predict_batch(self, samples: List[Dict], max_workers: int = 16) -> List[str]:
        # 【修改点】引入多线程并发
        def _safe_predict(s):
            try:
                return self.predict(s.get("sentence", ""), s.get("e1", ""), s.get("e2", ""))
            except Exception:
                return self.relation_names[-1]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_safe_predict, samples))


class VanillaRAG(BaseAgent):
    """Vanilla RAG 基线（与 PureLLM 接口一致）"""

    def __init__(self, relation_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.relation_names = relation_names or SEMEVAL_19

        rel_list = "\n".join(f"- {r}" for r in self.relation_names)
        template = (
            "判断以下句子中两个实体的关系。\n\n"
            "句子: {sentence}\n"
            "实体1: {e1}\n"
            "实体2: {e2}\n\n"
            "可选关系（只输出其中一个）:\n"
            + rel_list
            + "\n\n【强制要求】只输出关系名称，不要推理。\n\n"
            "你的回答: /no_think"
        )
        self.prompt = make_prompt(template, self.model_provider)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def predict(self, sentence: str, e1: str, e2: str) -> str:
        try:
            raw = self.chain.invoke({"sentence": sentence, "e1": e1, "e2": e2})
            return parse_relation_label(raw, valid_relations=self.relation_names)
        except Exception as e:
            print(f"VanillaRAG预测错误: {e}")
            return self.relation_names[-1]

    def predict_batch(self, samples: List[Dict], max_workers: int = 16) -> List[str]:
        # 【修改点】引入多线程并发
        def _safe_predict(s):
            try:
                return self.predict(s.get("sentence", ""), s.get("e1", ""), s.get("e2", ""))
            except Exception:
                return self.relation_names[-1]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_safe_predict, samples))


class FewShotPureLLM(BaseAgent):
    """Few-shot Pure LLM 基线"""

    def __init__(self, examples: List[Dict],
                 relation_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.examples = examples
        self.relation_names = relation_names or SEMEVAL_19

    def _format_prompt(self, sentence: str, e1: str, e2: str) -> str:
        example_text = ""
        for i, ex in enumerate(self.examples, 1):
            rel = ex.get("relation", "Other")
            ex_sent = ex.get("sentence", "")
            ex_e1 = ex.get("e1", "")
            ex_e2 = ex.get("e2", "")
            example_text += (
                f'{i}. 句子: "{ex_sent}" '
                f'[实体1: {ex_e1}; 实体2: {ex_e2}] → {rel}\n'
            )

        rel_list = "\n".join(f"  {i}. {r}" for i, r in enumerate(self.relation_names, 1))
        direction_hint = ""
        if any('(' in r for r in self.relation_names):
            direction_hint = "\n【注意】必须带 (e1,e2) 或 (e2,e1) 方向后缀。\n"

        prompt = (
            "判断以下句子中两个实体的关系。\n\n"
            "【参考示例】：\n"
            + example_text
            + "\n【待判断句子】：\n"
            f"句子: {sentence}\n实体1: {e1}\n实体2: {e2}\n\n"
            "【可选关系】（必须从以下列表中选择一个）：\n"
            + rel_list
            + direction_hint
            + "\n【强制要求】只输出关系名称，不要推理，不要解释。\n\n"
            "【你的回答】: /no_think"
        )
        return prompt


    def predict(self, sentence: str, e1: str, e2: str) -> str:
        try:
            prompt_str = self._format_prompt(sentence, e1, e2)
            raw = self.llm.invoke(prompt_str)
            if hasattr(raw, "content"):
                raw = raw.content
            return parse_relation_label(raw, valid_relations=self.relation_names)
        except Exception as e:
            print(f"FewShotPureLLM预测错误: {e}")
            return self.relation_names[-1]

    def predict_batch(self, samples: List[Dict], max_workers: int = 16) -> List[str]:
        # 【修改点】引入多线程并发
        def _safe_predict(s):
            try:
                return self.predict(s.get("sentence", ""), s.get("e1", ""), s.get("e2", ""))
            except Exception:
                return self.relation_names[-1]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_safe_predict, samples))


def run_baseline_comparison(samples: List[Dict],
                            model_provider: str = None,
                            relation_names: Optional[List[str]] = None) -> Dict:
    print("=" * 60)
    print("基线模型对比")
    print("=" * 60)
    print(f"样本数: {len(samples)}")

    models = {
        "Pure LLM": PureLLM(model_provider=model_provider, relation_names=relation_names),
        "Vanilla RAG": VanillaRAG(model_provider=model_provider, relation_names=relation_names),
    }

    results = {}
    for name, model in models.items():
        print(f"\n[运行] {name}...")
        preds = model.predict_batch(samples)
        acc = compute_accuracy(preds, [s.get("relation", "") for s in samples])
        results[name] = {"predictions": preds, "accuracy": acc}
        print(f"  {name} 准确率: {acc:.4f} ({acc * 100:.2f}%)")
    return results