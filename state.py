# from typing import List, Dict, Any, Optional, TypedDict
# from dataclasses import dataclass
#
#
# @dataclass
# class ExtractionResult:
#     """单个抽取结果"""
#     sentence: str
#     e1: str
#     e2: str
#     predicted_relation: str
#     confidence: float = 0.5
#     validation_passed: bool = False
#     validation_confidence: int = 50
#     validation_feedback: str = ""
#     refined_relation: str = ""
#     error: str = ""
#
#     def to_dict(self) -> Dict:
#         return {
#             "sentence": self.sentence,
#             "e1": self.e1,
#             "e2": self.e2,
#             "predicted_relation": self.predicted_relation,
#             "confidence": self.confidence,
#             "validation_passed": self.validation_passed,
#             "validation_confidence": self.validation_confidence,
#             "validation_feedback": self.validation_feedback,
#             "refined_relation": self.refined_relation,
#             "error": self.error
#         }
#
#
# class AgentState(TypedDict):
#     """多智能体共享状态"""
#     samples: List[Dict[str, Any]]
#     batch_size: int
#     extraction_results: List[ExtractionResult]
#     current_index: int
#     validation_scores: List[int]
#     validation_feedbacks: List[str]
#     final_predictions: List[str]
#     true_labels: List[str]
#     evaluation_metrics: Dict[str, float]
#     total_processed: int
#     total_errors: int
#     next_agent: str
#     error: Optional[str]
#
#
# def create_initial_state(samples: List[Dict], batch_size: int = 10) -> AgentState:
#     return AgentState(
#         samples=samples,
#         batch_size=batch_size,
#         extraction_results=[],
#         current_index=0,
#         validation_scores=[],
#         validation_feedbacks=[],
#         final_predictions=[],
#         true_labels=[],
#         evaluation_metrics={},
#         total_processed=0,
#         total_errors=0,
#         next_agent="extractor",
#         error=None
#     )

# state.py
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    sentence: str
    e1: str
    e2: str
    predicted_relation: str
    confidence: float = 0.5
    validation_passed: bool = False
    validation_confidence: int = 50
    validation_feedback: str = ""
    validation_suggested: str = ""     # ← 新增
    refined_relation: str = ""
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "sentence": self.sentence,
            "e1": self.e1,
            "e2": self.e2,
            "predicted_relation": self.predicted_relation,
            "confidence": self.confidence,
            "validation_passed": self.validation_passed,
            "validation_confidence": self.validation_confidence,
            "validation_feedback": self.validation_feedback,
            "validation_suggested": self.validation_suggested,  # ← 新增
            "refined_relation": self.refined_relation,
            "error": self.error,
        }


class AgentState(TypedDict):
    samples: List[Dict[str, Any]]
    batch_size: int
    extraction_results: List[ExtractionResult]
    current_index: int
    validation_scores: List[int]
    validation_feedbacks: List[str]
    validation_suggested: List[str]    # ← 新增
    final_predictions: List[str]
    true_labels: List[str]
    evaluation_metrics: Dict[str, float]
    total_processed: int
    total_errors: int
    next_agent: str
    error: Optional[str]


def create_initial_state(samples: List[Dict], batch_size: int = 10) -> AgentState:
    return AgentState(
        samples=samples,
        batch_size=batch_size,
        extraction_results=[],
        current_index=0,
        validation_scores=[],
        validation_feedbacks=[],
        validation_suggested=[],# ← 新增
        final_predictions=[],
        true_labels=[],
        evaluation_metrics={},
        total_processed=0,
        total_errors=0,
        next_agent="extractor",
        error=None,
    )