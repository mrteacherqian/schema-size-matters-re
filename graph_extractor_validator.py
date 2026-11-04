# graph_extractor_validator.py
"""
消融：Extractor + Validator（无 Refiner）

流程：Extractor → Validator → Evaluator

Validator 直接消费自己的输出做修正（不经过 Refiner）：
  - 如果 passed=False
  - 且 confidence >= conf_threshold（默认 85）
  - 且 suggested_relation != predicted_relation
  - 且 suggested_relation 在合法关系列表内
  → 用 suggested_relation 替换 Extractor 的预测
"""
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from state import AgentState, ExtractionResult, create_initial_state
from agents import ExtractorAgent, ValidatorAgent
from utils import set_seed, compute_accuracy, compute_macro_f1


class RelationExtractionGraphExtractorValidator:

    def __init__(self, model_provider: str = None,
                 temperature: float = None,
                 relation_names: Optional[List[str]] = None,
                 conf_threshold: int = 85):
        self.relation_names = relation_names
        self.conf_threshold = conf_threshold
        self._ev_triggered = 0        # ← 新增
        self._ev_changed = 0          # ← 新增
        self.extractor = ExtractorAgent(relation_names=relation_names,
                                        model_provider=model_provider,
                                        temperature=temperature)
        self.validator = ValidatorAgent(relation_names=relation_names,
                                        model_provider=model_provider,
                                        temperature=temperature)
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self):
        wf = StateGraph(AgentState)
        wf.add_node("extractor", self._extractor_node)
        wf.add_node("validator", self._validator_node)
        wf.add_node("evaluator", self._evaluator_node)
        wf.set_entry_point("extractor")
        wf.add_edge("extractor", "validator")
        wf.add_edge("validator", "evaluator")
        wf.add_edge("evaluator", END)
        return wf

    def _extractor_node(self, state: AgentState) -> AgentState:
        samples = state["samples"]
        idx = state["current_index"]
        results = state["extraction_results"]
        batch = state["batch_size"]

        start_idx = idx
        end_idx = min(idx + batch, len(samples))
        print(f"\n[E+V Extractor] 处理 {start_idx}-{end_idx - 1}")

        for i in range(start_idx, end_idx):
            s = samples[i]
            r = self.extractor.extract(s.get("sentence", ""),
                                       s.get("e1", ""), s.get("e2", ""))
            pred = r.get("predicted_relation", "Other")
            print(f"[E+V Extractor] #{i}: {pred}")
            results.append(ExtractionResult(
                sentence=r["sentence"], e1=r["e1"], e2=r["e2"],
                predicted_relation=pred,
                confidence=r.get("confidence", 0.5),
                error=r.get("error", ""),
            ))
            if "relation" in s:
                state["true_labels"].append(s["relation"])

        state["extraction_results"] = results
        state["current_index"] = end_idx
        state["next_agent"] = "validator"
        return state

    def _validator_node(self, state: AgentState) -> AgentState:
        results = state["extraction_results"]
        predictions, scores, feedbacks, suggested_list = [], [], [], []
        corrected = 0
        triggered = 0                                         # 新增
        print(f"\n[E+V Validator] 验证并修正 {len(results)} 条")
        for i, r in enumerate(results):
            v = self.validator.validate(r.sentence, r.e1, r.e2, r.predicted_relation)
            r.validation_passed = v.get("passed", False)
            r.validation_confidence = v.get("confidence", 50)
            r.validation_feedback = v.get("reason", "")
            r.validation_suggested = v.get("suggested_relation", r.predicted_relation)

            # E+V 修正规则
            final_rel = r.predicted_relation
            valid_suggested = (
                self.relation_names is None
                or r.validation_suggested in self.relation_names
            )
            if (not r.validation_passed
                    and r.validation_confidence >= self.conf_threshold
                    and r.validation_suggested
                    and r.validation_suggested != r.predicted_relation
                    and valid_suggested):
                triggered += 1                                # 新增
                final_rel = r.validation_suggested
                corrected += 1
                print(f"[E+V] #{i}: {r.predicted_relation} → {final_rel} "
                      f"(conf={r.validation_confidence})")

            r.refined_relation = final_rel
            predictions.append(final_rel)
            scores.append(r.validation_confidence)
            feedbacks.append(r.validation_feedback)
            suggested_list.append(r.validation_suggested)


        self._ev_triggered += triggered          # ← 跨 batch 累加
        self._ev_changed += corrected            # ← 跨 batch 累加
        print(f"[E+V] batch: triggered={triggered}/{len(results)}, "
              f"corrected={corrected}/{len(results)} | "
              f"cumulative: triggered={self._ev_triggered}, "
              f"corrected={self._ev_changed}")
        state["final_predictions"] = predictions
        state["validation_scores"] = scores
        state["validation_feedbacks"] = feedbacks
        state["validation_suggested"] = suggested_list
        state["next_agent"] = "evaluator"
        return state

    def _evaluator_node(self, state: AgentState) -> AgentState:
        preds = state["final_predictions"]
        labels = state["true_labels"]
        print(f"\n[E+V Evaluator] 评估 {len(preds)} 条")
        if preds and labels:
            acc = compute_accuracy(preds, labels)
            f1 = compute_macro_f1(preds, labels, self.relation_names)
            print(f"[E+V Evaluator] Acc={acc:.4f} | Macro-F1={f1['macro_f1']:.4f}")
            state["evaluation_metrics"] = {"accuracy": acc, "macro_f1": f1["macro_f1"]}
        else:
            state["evaluation_metrics"] = {"accuracy": 0.0, "macro_f1": 0.0}
        state["next_agent"] = "end"
        return state

    def run(self, samples: List[Dict], batch_size: int = 10) -> Dict[str, Any]:
        set_seed()
        self._ev_triggered = 0                   # ← 新增：每次 run 重置
        self._ev_changed = 0
        if not samples:
            return {"predictions": [], "metrics": {}, "success": False}
        all_predictions = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            init = create_initial_state(batch, batch_size)
            final = self.app.invoke(init)
            all_predictions.extend(final.get("final_predictions", []))

        labels = [s.get("relation", "") for s in samples]
        acc = compute_accuracy(all_predictions, labels)
        f1 = compute_macro_f1(all_predictions, labels, self.relation_names)
        return {
            "predictions": all_predictions,
            "metrics": {"accuracy": acc, "macro_f1": f1["macro_f1"],
                        "num_samples": len(all_predictions),
                        "ev_triggered": self._ev_triggered,  # ← 放进 metrics 内部
                        "ev_changed": self._ev_changed,
                        },

            "success": True,
        }