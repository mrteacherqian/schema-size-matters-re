# # graph_extractor_only.py - 纯抽取器（消融实验）
# from typing import List, Dict, Any
# from langgraph.graph import StateGraph, END
#
# from state import AgentState, ExtractionResult, create_initial_state
# from agents import ExtractorAgent
# from utils import set_seed, compute_accuracy
#
#
# class RelationExtractionGraphExtractorOnly:
#     """消融实验：纯抽取器（无验证器、无精炼器）"""
#
#     def __init__(self, model_provider: str = None, temperature: float = None):
#         self.extractor = ExtractorAgent(model_provider=model_provider, temperature=temperature)
#         self.graph = self._build_graph()
#         self.app = self.graph.compile()
#
#     def _build_graph(self):
#         workflow = StateGraph(AgentState)
#         workflow.add_node("extractor", self._extractor_node)
#         workflow.add_node("evaluator", self._evaluator_node)
#         workflow.set_entry_point("extractor")
#         workflow.add_edge("extractor", "evaluator")
#         workflow.add_edge("evaluator", END)
#         return workflow
#
#     def _extractor_node(self, state: AgentState) -> AgentState:
#         samples, idx, results = state["samples"], state["current_index"], state["extraction_results"]
#         batch = state["batch_size"]
#
#         start_idx = idx
#         end_idx = min(idx + batch, len(samples))
#         print(f"\n[纯抽取器] 处理样本 {start_idx} 到 {end_idx - 1}")
#
#         for i in range(start_idx, end_idx):
#             s = samples[i]
#             r = self.extractor.extract(s.get("sentence", ""), s.get("e1", ""), s.get("e2", ""))
#             pred = r.get("predicted_relation", "Other")
#             print(f"[纯抽取器] 样本 {i}: 预测={pred}")
#
#             results.append(ExtractionResult(
#                 sentence=r["sentence"], e1=r["e1"], e2=r["e2"],
#                 predicted_relation=pred,
#                 confidence=r.get("confidence", 0.5),
#                 error=r.get("error", "")
#             ))
#
#             if "relation" in s:
#                 state["true_labels"].append(s["relation"])
#
#         state["extraction_results"] = results
#         state["current_index"] = end_idx
#         state["total_processed"] = len(results)
#         state["next_agent"] = "evaluator"
#         return state
#
#     def _evaluator_node(self, state: AgentState) -> AgentState:
#         preds = [r.predicted_relation for r in state["extraction_results"]]
#         labels = state["true_labels"]
#
#         # ===== 关键修复：先存入 final_predictions =====
#         state["final_predictions"] = preds
#         # ===========================================
#
#         if preds and labels:
#             acc = compute_accuracy(preds, labels)
#             print(f"[Evaluator] 准确率: {acc:.4f} ({acc * 100:.2f}%)")
#             state["evaluation_metrics"] = {"accuracy": acc, "num_samples": len(preds)}
#         else:
#             state["evaluation_metrics"] = {"accuracy": 0.0, "num_samples": 0}
#
#         state["next_agent"] = "end"
#         return state
#
#     def run(self, samples: List[Dict], batch_size: int = 10) -> Dict[str, Any]:
#         set_seed()
#         if not samples:
#             return {"predictions": [], "metrics": {}, "success": False}
#
#         all_predictions = []
#         for start_idx in range(0, len(samples), batch_size):
#             batch_samples = samples[start_idx:start_idx + batch_size]
#             print(f"\n[Graph] 批次 {start_idx // batch_size + 1}")
#             final = self.app.invoke(create_initial_state(batch_samples, batch_size))
#             all_predictions.extend(final.get("final_predictions", []))
#
#         true_labels = [s.get("relation", "") for s in samples]
#         overall_acc = compute_accuracy(all_predictions, true_labels)
#
#         return {
#             "predictions": all_predictions,
#             "metrics": {"accuracy": overall_acc, "num_samples": len(all_predictions)},
#             "success": True
#         }


# graph_extractor_only.py
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from state import AgentState, ExtractionResult, create_initial_state
from agents import ExtractorAgent
from utils import set_seed, compute_accuracy, compute_macro_f1


class RelationExtractionGraphExtractorOnly:
    """消融：纯抽取器"""

    def __init__(self, model_provider: str = None,
                 temperature: float = None,
                 relation_names: Optional[List[str]] = None):
        self.relation_names = relation_names
        self.extractor = ExtractorAgent(relation_names=relation_names,
                                        model_provider=model_provider,
                                        temperature=temperature)
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self):
        wf = StateGraph(AgentState)
        wf.add_node("extractor", self._extractor_node)
        wf.add_node("evaluator", self._evaluator_node)
        wf.set_entry_point("extractor")
        wf.add_edge("extractor", "evaluator")
        wf.add_edge("evaluator", END)
        return wf

    def _extractor_node(self, state: AgentState) -> AgentState:
        samples = state["samples"]
        idx = state["current_index"]
        results = state["extraction_results"]
        batch = state["batch_size"]

        start_idx = idx
        end_idx = min(idx + batch, len(samples))
        print(f"\n[ExtractorOnly] 处理 {start_idx}-{end_idx - 1}")

        for i in range(start_idx, end_idx):
            s = samples[i]
            r = self.extractor.extract(s.get("sentence", ""),
                                       s.get("e1", ""), s.get("e2", ""))
            pred = r.get("predicted_relation", "Other")
            print(f"[ExtractorOnly] #{i}: {pred}")
            results.append(ExtractionResult(
                sentence=r["sentence"], e1=r["e1"], e2=r["e2"],
                predicted_relation=pred, confidence=r.get("confidence", 0.5),
                error=r.get("error", ""),
            ))
            if "relation" in s:
                state["true_labels"].append(s["relation"])

        state["extraction_results"] = results
        state["current_index"] = end_idx
        state["next_agent"] = "evaluator"
        return state

    def _evaluator_node(self, state: AgentState) -> AgentState:
        preds = [r.predicted_relation for r in state["extraction_results"]]
        labels = state["true_labels"]
        state["final_predictions"] = preds

        if preds and labels:
            acc = compute_accuracy(preds, labels)
            f1 = compute_macro_f1(preds, labels, self.relation_names)
            print(f"[Evaluator] Acc={acc:.4f} | Macro-F1={f1['macro_f1']:.4f}")
            state["evaluation_metrics"] = {"accuracy": acc, "macro_f1": f1["macro_f1"]}
        else:
            state["evaluation_metrics"] = {"accuracy": 0.0, "macro_f1": 0.0}
        state["next_agent"] = "end"
        return state

    def run(self, samples: List[Dict], batch_size: int = 10) -> Dict[str, Any]:
        set_seed()
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
                        "num_samples": len(all_predictions)},
            "success": True,
        }