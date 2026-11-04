#
#
# # graph.py
# from typing import List, Dict, Any, Optional
# from langgraph.graph import StateGraph, END
#
# from state import AgentState, ExtractionResult, create_initial_state
# from agents import ExtractorAgent, ValidatorAgent, RefinerAgent
# from utils import set_seed, compute_accuracy, compute_macro_f1
#
#
# class RelationExtractionGraph:
#     """多智能体协作图：Extractor → Validator → Refiner → Evaluator"""
#
#     def __init__(self, model_provider: str = None,
#                  temperature: float = None,
#                  relation_names: Optional[List[str]] = None):
#         self.relation_names = relation_names
#         self.extractor = ExtractorAgent(relation_names=relation_names,
#                                         model_provider=model_provider,
#                                         temperature=temperature)
#         self.validator = ValidatorAgent(relation_names=relation_names,
#                                         model_provider=model_provider,
#                                         temperature=temperature)
#         self.refiner = RefinerAgent(relation_names=relation_names,
#                                     model_provider=model_provider,
#                                     temperature=temperature)
#         self.graph = self._build_graph()
#         self.app = self.graph.compile()
#
#     def _build_graph(self):
#         wf = StateGraph(AgentState)
#         wf.add_node("extractor", self._extractor_node)
#         wf.add_node("validator", self._validator_node)
#         wf.add_node("refiner", self._refiner_node)
#         wf.add_node("evaluator", self._evaluator_node)
#         wf.set_entry_point("extractor")
#         wf.add_edge("extractor", "validator")
#         wf.add_edge("validator", "refiner")
#         wf.add_edge("refiner", "evaluator")
#         wf.add_edge("evaluator", END)
#         return wf
#
#     def _extractor_node(self, state: AgentState) -> AgentState:
#         samples = state["samples"]
#         idx = state["current_index"]
#         results = state["extraction_results"]
#         batch = state["batch_size"]
#
#         start_idx = idx
#         end_idx = min(idx + batch, len(samples))
#         print(f"\n[Extractor] 处理 {start_idx}-{end_idx - 1}")
#
#         for i in range(start_idx, end_idx):
#             s = samples[i]
#             r = self.extractor.extract(s.get("sentence", ""),
#                                        s.get("e1", ""), s.get("e2", ""))
#             pred = r.get("predicted_relation", "Other")
#             print(f"[Extractor] #{i}: {pred}")
#
#             results.append(ExtractionResult(
#                 sentence=r["sentence"], e1=r["e1"], e2=r["e2"],
#                 predicted_relation=pred,
#                 confidence=r.get("confidence", 0.5),
#                 error=r.get("error", ""),
#             ))
#             if "relation" in s:
#                 state["true_labels"].append(s["relation"])
#
#         state["extraction_results"] = results
#         state["current_index"] = end_idx
#         state["total_processed"] = len(results)
#         state["next_agent"] = "validator"
#         return state
#
#     def _validator_node(self, state: AgentState) -> AgentState:
#         results = state["extraction_results"]
#         scores, feedbacks = [], []
#         suggested_list = []  # ← 新增
#         print(f"\n[Validator] 验证 {len(results)} 条")
#         for i, r in enumerate(results):
#             v = self.validator.validate(r.sentence, r.e1, r.e2, r.predicted_relation)
#             r.validation_passed = v.get("passed", False)
#             r.validation_confidence = v.get("confidence", 50)
#             r.validation_feedback = v.get("reason", "")
#             r.validation_suggested = v.get("suggested_relation", r.predicted_relation)  # 新增
#             if not r.validation_passed:
#                 r.confidence *= 0.5
#
#             print(f"[Validator] #{i}: passed={r.validation_passed}, conf={r.validation_confidence}")
#             scores.append(r.validation_confidence)
#             feedbacks.append(r.validation_feedback)
#             suggested_list.append(r.validation_suggested)  # ← 新增
#         state["validation_scores"] = scores
#         state["validation_feedbacks"] = feedbacks
#         state["validation_suggested"] = suggested_list  # ← 新增
#         state["next_agent"] = "refiner"
#         return state
#
#     def _refiner_node(self, state: AgentState) -> AgentState:
#         results = state["extraction_results"]
#         scores = state["validation_scores"]
#         feedbacks = state["validation_feedbacks"]
#         suggested = state["validation_suggested"]
#         predictions = []
#         print(f"\n[Refiner] 精炼 {len(results)} 条")
#         refined_count = 0
#         for i, r in enumerate(results):
#             v_score = scores[i] if i < len(scores) else 50
#             v_feedback = feedbacks[i] if i < len(feedbacks) else ""
#             v_suggested = suggested[i] if i < len(suggested) else r.predicted_relation
#             # # 触发 Refiner 的条件：
#             # # 1. 验证器置信度 >= 70（原有）
#             # # 2. 验证器未通过且建议了不同关系（新增，覆盖 conf=60 但 passed=False 的情况）
#             # should_refine = (
#             #         v_score >= 70
#             #         or (not r.validation_passed and v_suggested and v_suggested != r.predicted_relation)
#             # )
#             # ===== 收紧后的触发条件 =====
#             # 只有满足以下任一才触发 Refiner：
#             #   1. Validator 明确反对（passed=False）且高置信度（>= 85）
#             #   2. Validator 建议了不同关系且极高置信度（>= 90）
#             should_refine = (
#                     (not r.validation_passed and v_score >= 85)
#                     or (v_suggested != r.predicted_relation and v_score >= 90 and v_suggested)
#             )
#             # ==============================
#             if should_refine:
#                 refined = self.refiner.refine(
#                     r.sentence, r.e1, r.e2,
#                     r.predicted_relation, v_feedback, v_score,
#                     v_suggested,  # ← 新增
#                 )
#                 final_rel = refined.get("final_relation", r.predicted_relation)
#                 if final_rel != r.predicted_relation:
#                     refined_count += 1
#                     print(f"[Refiner] #{i}: {r.predicted_relation} → {final_rel} "
#                           f"(conf={v_score}, passed={r.validation_passed}, suggested={v_suggested})")
#             else:
#                 final_rel = r.predicted_relation
#
#
#             r.refined_relation = final_rel
#             predictions.append(final_rel)
#
#         print(f"[Refiner] 共精炼 {refined_count}/{len(results)} 条")
#         state["final_predictions"] = predictions
#         state["next_agent"] = "evaluator"
#         return state
#
#     def _evaluator_node(self, state: AgentState) -> AgentState:
#         preds = state["final_predictions"]
#         labels = state["true_labels"]
#         print(f"\n[Evaluator] 评估 {len(preds)} 条")
#
#         if preds and labels:
#             acc = compute_accuracy(preds, labels)
#             f1_result = compute_macro_f1(preds, labels, self.relation_names)
#             print(f"[Evaluator] Acc={acc:.4f} | Macro-F1={f1_result['macro_f1']:.4f}")
#             state["evaluation_metrics"] = {
#                 "accuracy": acc,
#                 "macro_f1": f1_result["macro_f1"],
#                 "num_samples": len(preds),
#                 "num_errors": state.get("total_errors", 0),
#             }
#         else:
#             state["evaluation_metrics"] = {
#                 "accuracy": 0.0, "macro_f1": 0.0,
#                 "num_samples": 0, "num_errors": 0,
#             }
#         state["next_agent"] = "end"
#         return state
#
#     def run(self, samples: List[Dict], batch_size: int = 10) -> Dict[str, Any]:
#         set_seed()
#         if not samples:
#             return {"predictions": [], "metrics": {}, "success": False, "error": "No samples"}
#
#         all_predictions = []
#         all_results = []
#         total_errors = 0
#
#         for start_idx in range(0, len(samples), batch_size):
#             batch_samples = samples[start_idx:start_idx + batch_size]
#             print(f"\n[Graph] 批次 {start_idx // batch_size + 1}")
#             try:
#                 init = create_initial_state(batch_samples, batch_size)
#                 final = self.app.invoke(init)
#                 all_predictions.extend(final.get("final_predictions", []))
#                 all_results.extend([r.to_dict() for r in final.get("extraction_results", [])])
#                 total_errors += final.get("total_errors", 0)
#             except Exception as e:
#                 print(f"[Graph] 批次失败: {e}")
#                 continue
#
#         true_labels = [s.get("relation", "") for s in samples]
#         acc = compute_accuracy(all_predictions, true_labels)
#         f1_result = compute_macro_f1(all_predictions, true_labels, self.relation_names)
#
#         return {
#             "predictions": all_predictions,
#             "metrics": {
#                 "accuracy": acc,
#                 "macro_f1": f1_result["macro_f1"],
#                 "num_samples": len(all_predictions),
#                 "num_errors": total_errors,
#             },
#             "detailed_results": all_results,
#             "success": True,
#         }

# graph.py
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END

from state import AgentState, ExtractionResult, create_initial_state
from agents import ExtractorAgent, ValidatorAgent, RefinerAgent
from utils import set_seed, compute_accuracy, compute_macro_f1


class RelationExtractionGraph:
    """多智能体协作图：Extractor → Validator → Refiner → Evaluator"""

    def __init__(self, model_provider: str = None,
                 temperature: float = None,
                 relation_names: Optional[List[str]] = None,
                 conf_threshold: int = 85,          # 新增
                 suggest_threshold: int = 90):      # 新增
        self.relation_names = relation_names
        self.conf_threshold = conf_threshold
        self.suggest_threshold = suggest_threshold
        self._refiner_triggered = 0                     # 新增
        self._refiner_changed = 0                       # 新增
        self.extractor = ExtractorAgent(relation_names=relation_names,
                                        model_provider=model_provider,
                                        temperature=temperature)
        self.validator = ValidatorAgent(relation_names=relation_names,
                                        model_provider=model_provider,
                                        temperature=temperature)
        self.refiner = RefinerAgent(relation_names=relation_names,
                                    model_provider=model_provider,
                                    temperature=temperature)
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self):
        wf = StateGraph(AgentState)
        wf.add_node("extractor", self._extractor_node)
        wf.add_node("validator", self._validator_node)
        wf.add_node("refiner", self._refiner_node)
        wf.add_node("evaluator", self._evaluator_node)
        wf.set_entry_point("extractor")
        wf.add_edge("extractor", "validator")
        wf.add_edge("validator", "refiner")
        wf.add_edge("refiner", "evaluator")
        wf.add_edge("evaluator", END)
        return wf

    def _extractor_node(self, state: AgentState) -> AgentState:
        samples = state["samples"]
        idx = state["current_index"]
        results = state["extraction_results"]
        batch = state["batch_size"]

        start_idx = idx
        end_idx = min(idx + batch, len(samples))
        print(f"\n[Extractor] 处理 {start_idx}-{end_idx - 1} (并发)")

        def _extract_one(i):
            s = samples[i]
            return self.extractor.extract(s.get("sentence", ""),
                                          s.get("e1", ""), s.get("e2", ""))

        # 【修改点】并发跑当前 batch
        with ThreadPoolExecutor(max_workers=batch) as executor:
            extract_results = list(executor.map(_extract_one, range(start_idx, end_idx)))

        for i, r in zip(range(start_idx, end_idx), extract_results):
            pred = r.get("predicted_relation", "Other")
            print(f"[Extractor] #{i}: {pred}")
            results.append(ExtractionResult(
                sentence=r["sentence"], e1=r["e1"], e2=r["e2"],
                predicted_relation=pred,
                confidence=r.get("confidence", 0.5),
                error=r.get("error", ""),
            ))
            if "relation" in samples[i]:
                state["true_labels"].append(samples[i]["relation"])

        state["extraction_results"] = results
        state["current_index"] = end_idx
        state["total_processed"] = len(results)
        state["next_agent"] = "validator"
        return state

    def _validator_node(self, state: AgentState) -> AgentState:
        results = state["extraction_results"]
        scores, feedbacks = [], []
        suggested_list = []
        print(f"\n[Validator] 验证 {len(results)} 条 (并发)")

        def _validate_one(r):
            return self.validator.validate(r.sentence, r.e1, r.e2, r.predicted_relation)

        # 【修改点】并发执行 Validator
        with ThreadPoolExecutor(max_workers=len(results)) as executor:
            v_results = list(executor.map(_validate_one, results))

        for i, (r, v) in enumerate(zip(results, v_results)):
            r.validation_passed = v.get("passed", False)
            r.validation_confidence = v.get("confidence", 50)
            r.validation_feedback = v.get("reason", "")
            r.validation_suggested = v.get("suggested_relation", r.predicted_relation)
            if not r.validation_passed:
                r.confidence *= 0.5

            print(f"[Validator] #{i}: passed={r.validation_passed}, conf={r.validation_confidence}")
            scores.append(r.validation_confidence)
            feedbacks.append(r.validation_feedback)
            suggested_list.append(r.validation_suggested)

        state["validation_scores"] = scores
        state["validation_feedbacks"] = feedbacks
        state["validation_suggested"] = suggested_list
        state["next_agent"] = "refiner"
        return state

    # def _refiner_node(self, state: AgentState) -> AgentState:
    #     results = state["extraction_results"]
    #     scores = state["validation_scores"]
    #     feedbacks = state["validation_feedbacks"]
    #     suggested = state["validation_suggested"]
    #     predictions = [None] * len(results)
    #     print(f"\n[Refiner] 精炼 {len(results)} 条 (并发)")
    #
    #     def _refine_one(i):
    #         r = results[i]
    #         v_score = scores[i] if i < len(scores) else 50
    #         v_feedback = feedbacks[i] if i < len(feedbacks) else ""
    #         v_suggested = suggested[i] if i < len(suggested) else r.predicted_relation
    #
    #         should_refine = (
    #                 (not r.validation_passed and v_score >= 85) ####零时改成60阈值，看效果，原值是85
    #                 or (v_suggested != r.predicted_relation and v_score >= 90 and v_suggested)###原址90
    #         )
    #         # should_refine = (
    #         #         (not r.validation_passed and v_score >= 60) ####零时改成60阈值，看效果，原值是85
    #         #         or (v_suggested != r.predicted_relation and v_score >= 80 and v_suggested)###原址90
    #         # )
    #         if should_refine:
    #             refined = self.refiner.refine(
    #                 r.sentence, r.e1, r.e2,
    #                 r.predicted_relation, v_feedback, v_score,
    #                 v_suggested,
    #             )
    #             final_rel = refined.get("final_relation", r.predicted_relation)
    #             if final_rel != r.predicted_relation:
    #                 print(f"[Refiner] #{i}: {r.predicted_relation} → {final_rel} "
    #                       f"(conf={v_score}, passed={r.validation_passed}, suggested={v_suggested})")
    #             return final_rel
    #         return r.predicted_relation
    #
    #     # 【修改点】并发执行 Refiner
    #     with ThreadPoolExecutor(max_workers=len(results)) as executor:
    #         final_rels = list(executor.map(_refine_one, range(len(results))))
    #
    #     for i, final_rel in enumerate(final_rels):
    #         results[i].refined_relation = final_rel
    #         predictions[i] = final_rel
    #
    #     state["final_predictions"] = predictions
    #     state["next_agent"] = "evaluator"
    #     return state
    def _refiner_node(self, state: AgentState) -> AgentState:
        results = state["extraction_results"]
        scores = state["validation_scores"]
        feedbacks = state["validation_feedbacks"]
        suggested = state["validation_suggested"]
        predictions = [None] * len(results)
        print(f"\n[Refiner] 精炼 {len(results)} 条 (并发)")
        triggered = []       # list 代替 int，避免 ThreadPoolExecutor 竞态

        def _refine_one(i):
            r = results[i]
            v_score = scores[i] if i < len(scores) else 50
            v_feedback = feedbacks[i] if i < len(feedbacks) else ""
            v_suggested = suggested[i] if i < len(suggested) else r.predicted_relation

            should_refine = (
                    (not r.validation_passed and v_score >= self.conf_threshold)
                    or (v_suggested != r.predicted_relation
                        and v_score >= self.suggest_threshold and v_suggested)
            )
            if should_refine:
                triggered.append(1)
                refined = self.refiner.refine(
                    r.sentence, r.e1, r.e2,
                    r.predicted_relation, v_feedback, v_score, v_suggested,
                )
                final_rel = refined.get("final_relation", r.predicted_relation)
                if final_rel != r.predicted_relation:
                    print(f"[Refiner] #{i}: {r.predicted_relation} → {final_rel} "
                          f"(conf={v_score}, passed={r.validation_passed}, suggested={v_suggested})")
                    return final_rel, True
                return final_rel, False
            return r.predicted_relation, False

        with ThreadPoolExecutor(max_workers=len(results)) as executor:
            outs = list(executor.map(_refine_one, range(len(results))))

        changed = 0
        for i, (final_rel, chg) in enumerate(outs):
            results[i].refined_relation = final_rel
            predictions[i] = final_rel
            changed += int(chg)

        self._refiner_triggered += len(triggered)
        self._refiner_changed += changed
        print(f"[Refiner] batch: triggered={len(triggered)}/{len(results)}, "
              f"changed={changed} | cumulative: triggered={self._refiner_triggered}, "
              f"changed={self._refiner_changed}")

        state["final_predictions"] = predictions
        state["next_agent"] = "evaluator"
        return state
    def _evaluator_node(self, state: AgentState) -> AgentState:
        preds = state["final_predictions"]
        labels = state["true_labels"]
        print(f"\n[Evaluator] 评估 {len(preds)} 条")

        if preds and labels:
            acc = compute_accuracy(preds, labels)
            f1_result = compute_macro_f1(preds, labels, self.relation_names)
            print(f"[Evaluator] Acc={acc:.4f} | Macro-F1={f1_result['macro_f1']:.4f}")
            state["evaluation_metrics"] = {
                "accuracy": acc,
                "macro_f1": f1_result["macro_f1"],
                "num_samples": len(preds),
                "num_errors": state.get("total_errors", 0),
            }
        else:
            state["evaluation_metrics"] = {
                "accuracy": 0.0, "macro_f1": 0.0,
                "num_samples": 0, "num_errors": 0,
            }
        state["next_agent"] = "end"
        return state

    def run(self, samples: List[Dict], batch_size: int = 10) -> Dict[str, Any]:
        # set_seed()
        # if not samples:
        #     return {"predictions": [], "metrics": {}, "success": False, "error": "No samples"}
        set_seed()
        self._refiner_triggered = 0
        self._refiner_changed = 0
        if not samples:
            return {"predictions": [], "metrics": {}, "success": False, "error": "No samples"}

        all_predictions = []
        all_results = []
        total_errors = 0

        # 为了防止内存爆掉，每次还是处理一个 batch
        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx:start_idx + batch_size]
            print(f"\n[Graph] 批次 {start_idx // batch_size + 1}")
            try:
                init = create_initial_state(batch_samples, batch_size)
                final = self.app.invoke(init)
                all_predictions.extend(final.get("final_predictions", []))
                all_results.extend([r.to_dict() for r in final.get("extraction_results", [])])
                total_errors += final.get("total_errors", 0)
            except Exception as e:
                print(f"[Graph] 批次失败: {e}")
                continue

        true_labels = [s.get("relation", "") for s in samples]
        acc = compute_accuracy(all_predictions, true_labels)
        f1_result = compute_macro_f1(all_predictions, true_labels, self.relation_names)

        return {
            "predictions": all_predictions,
            "metrics": {
                "accuracy": acc,
                "macro_f1": f1_result["macro_f1"],
                "num_samples": len(all_predictions),
                "num_errors": total_errors,
                "refiner_triggered": self._refiner_triggered,
                "refiner_changed": self._refiner_changed,
            },
            "detailed_results": all_results,
            "success": True,
        }
