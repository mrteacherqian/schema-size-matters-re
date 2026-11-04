# import json
# import re
# import os
# import random
# from typing import List, Dict, Any, Tuple, Optional
# import numpy as np
#
#
# def set_seed(seed: int = 42):
#     """设置随机种子"""
#     random.seed(seed)
#     np.random.seed(seed)
#
#
# def ensure_dir(path: str):
#     """确保目录存在"""
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#
# def save_json(data: Any, filepath: str, indent: int = 2):
#     """保存JSON"""
#     ensure_dir(os.path.dirname(filepath))
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=indent)
#
#
# def load_json(filepath: str) -> Any:
#     """加载JSON"""
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
#
# def extract_entities_from_text(text: str) -> Tuple[str, str, str]:
#     """从带<e1><e2>标记的文本中提取实体"""
#     e1_match = re.search(r'<e1>(.*?)</e1>', text)
#     e2_match = re.search(r'<e2>(.*?)</e2>', text)
#     e1 = e1_match.group(1) if e1_match else ""
#     e2 = e2_match.group(1) if e2_match else ""
#     clean_text = re.sub(r'</?e1>|</?e2>', '', text)
#     return clean_text, e1, e2
#
#
# # utils.py - 替换 parse_relation_label 函数
#
# def parse_relation_label(response: str) -> str:
#     """
#     解析LLM输出的关系标签（提取第一个出现的关系名）
#     """
#     if not response:
#         return "Other"
#
#     response = response.strip()
#
#     # 关系名列表（按长度降序排列，优先匹配长词）
#     all_relations = [
#         'Cause-Effect', 'Component-Whole', 'Content-Container',
#         'Entity-Destination', 'Entity-Origin', 'Instrument-Agency',
#         'Member-Collection', 'Message-Topic', 'Product-Producer',
#         'Other'
#     ]
#
#     # 1. 先尝试匹配 "RelationName(e1,e2)" 格式
#     pattern = r'([A-Za-z\-]+)\s*\(\s*e1\s*,\s*e2\s*\)'
#     match = re.search(pattern, response, re.IGNORECASE)
#     if match:
#         return match.group(1)
#
#     # 2. 提取所有出现的关系名，取第一个
#     found = []
#     for rel in sorted(all_relations, key=len, reverse=True):
#         if rel.lower() in response.lower():
#             found.append(rel)
#
#     # 如果有匹配，返回第一个（按出现顺序，不是按长度）
#     if found:
#         # 按在原文中首次出现的位置排序
#         positions = []
#         for rel in found:
#             pos = response.lower().find(rel.lower())
#             positions.append((pos, rel))
#         # 按位置排序，取最早出现的
#         positions.sort(key=lambda x: x[0])
#         return positions[0][1]
#
#     # 3. 中文映射
#     chinese_map = {
#         '因果关系': 'Cause-Effect',
#         '原因': 'Cause-Effect',
#         '组成关系': 'Component-Whole',
#         '组成': 'Component-Whole',
#         '内容': 'Content-Container',
#         '目的地': 'Entity-Destination',
#         '来源': 'Entity-Origin',
#         '工具': 'Instrument-Agency',
#         '成员': 'Member-Collection',
#         '集合': 'Member-Collection',
#         '主题': 'Message-Topic',
#         '信息': 'Message-Topic',
#         '生产者': 'Product-Producer',
#         '产品': 'Product-Producer',
#     }
#     for cn, en in chinese_map.items():
#         if cn in response:
#             return en
#
#     return "Other"
#
#
#
# # def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
# #     """
# #     计算准确率 - 从真实标签中提取关系名再比较
# #     支持 "Product-Producer(e1,e2)" 或 "Product-Producer(e2,e1)" 格式
# #     """
# #     if len(predictions) != len(labels) or len(predictions) == 0:
# #         return 0.0
#
# #     import re
# #     correct = 0
# #     for pred, label in zip(predictions, labels):
# #         # 从标签中提取关系名（如 "Product-Producer(e2,e1)" -> "Product-Producer"）
# #         match = re.match(r'([A-Za-z\-]+)', label)
# #         if match:
# #             true_relation = match.group(1)
# #         else:
# #             true_relation = label
#
# #         # 比较（忽略大小写）
# #         if pred.lower() == true_relation.lower():
# #             correct += 1
#
# #     return correct / len(predictions)
# ##只计算关系名不管方向的，准确率会提高
# def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
#     """
#     计算准确率 - 只比较关系名，忽略方向
#     """
#     if len(predictions) != len(labels) or len(predictions) == 0:
#         return 0.0
#
#     import re
#     correct = 0
#     for pred, label in zip(predictions, labels):
#         # 从预测中提取关系名
#         pred_match = re.match(r'([A-Za-z\-]+)', pred)
#         pred_name = pred_match.group(1) if pred_match else pred
#
#         # 从真实标签中提取关系名
#         label_match = re.match(r'([A-Za-z\-]+)', label)
#         true_name = label_match.group(1) if label_match else label
#
#         if pred_name.lower() == true_name.lower():
#             correct += 1
#
#     return correct / len(predictions)
#
# def compute_f1(predictions: List[str], labels: List[str],
#                relation_names: List[str]) -> Dict[str, float]:
#     """计算每个关系的F1和宏平均F1"""
#     # 构建混淆矩阵统计
#     tp = {rel: 0 for rel in relation_names}
#     fp = {rel: 0 for rel in relation_names}
#     fn = {rel: 0 for rel in relation_names}
#
#     for pred, true in zip(predictions, labels):
#         if pred == true:
#             tp[pred] = tp.get(pred, 0) + 1
#         else:
#             fp[pred] = fp.get(pred, 0) + 1
#             fn[true] = fn.get(true, 0) + 1
#
#     # 计算每个关系的F1
#     f1s = {}
#     for rel in relation_names:
#         p = tp[rel] / (tp[rel] + fp[rel]) if (tp[rel] + fp[rel]) > 0 else 0
#         r = tp[rel] / (tp[rel] + fn[rel]) if (tp[rel] + fn[rel]) > 0 else 0
#         f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
#         f1s[rel] = f1
#
#     # 宏平均F1
#     macro_f1 = sum(f1s.values()) / len(f1s) if f1s else 0
#
#     return {"macro_f1": macro_f1, "per_relation": f1s}


# utils.py
import json
import re
import os
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


# ========== SemEval-2010 官方 19 类（双向）==========
SEMEVAL_19 = [
    'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
    'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
    'Content-Container(e1,e2)', 'Content-Container(e2,e1)',
    'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
    'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
    'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
    'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
    'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
    'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
    'Other',
]

# ========== SemEval-2010 无方向 10 类（备用）==========
SEMEVAL_10 = [
    'Cause-Effect', 'Component-Whole', 'Content-Container',
    'Entity-Destination', 'Entity-Origin', 'Instrument-Agency',
    'Member-Collection', 'Message-Topic', 'Product-Producer', 'Other'
]

# ========== SciERC 关系类型 与诗句数据集一致==========
SCIERC_RELATIONS = [
    'Used-for', 'Evaluate-for', 'Feature-of', 'Part-of',
    'Compare', 'Conjunction', 'Hyponym-of',
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(data: Any, filepath: str, indent: int = 2):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entities_from_text(text: str) -> Tuple[str, str, str]:
    """从带 <e1></e1><e2></e2> 标记的文本中提取实体"""
    e1_match = re.search(r'<e1>(.*?)</e1>', text)
    e2_match = re.search(r'<e2>(.*?)</e2>', text)
    e1 = e1_match.group(1) if e1_match else ""
    e2 = e2_match.group(1) if e2_match else ""
    clean_text = re.sub(r'</?e1>|</?e2>', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text, e1, e2


# ========== 标签解析 ==========

# def _normalize_label(label: str) -> str:
#     """标准化标签：统一方向标记与大小写"""
#     label = label.strip()
#
#     m = re.match(r'([A-Za-z\-_]+)\s*\(\s*e1\s*,\s*e2\s*\)', label, re.IGNORECASE)
#     if m:
#         return f"{m.group(1)}(e1,e2)"
#     m = re.match(r'([A-Za-z\-_]+)\s*\(\s*e2\s*,\s*e1\s*\)', label, re.IGNORECASE)
#     if m:
#         return f"{m.group(1)}(e2,e1)"
#     return label

def _normalize_label(label: str) -> str:
    """标准化标签：统一方向标记与大小写"""
    label = label.strip()

    m = re.match(r'([A-Za-z\-_]+)\s*\(\s*e1\s*,\s*e2\s*\)', label, re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}(e1,e2)"
    m = re.match(r'([A-Za-z\-_]+)\s*\(\s*e2\s*,\s*e1\s*\)', label, re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}(e2,e1)"

    # 无方向关系名：统一小写
    return label.lower()



def parse_relation_label(response: str,
                        valid_relations: Optional[List[str]] = None) -> str:
    """
    从 LLM 输出中提取关系名。
    优先顺序：
      1. "答案:" / "最终答案:" / "your answer:" 之后的关系名
      2. 全文中最后出现的关系名
      3. 兜底 "Other"
    """
    # ---- 1. 默认值 ----
    if valid_relations is None:
        valid_relations = SEMEVAL_10

    # ---- 2. 空输入兜底 ----
    if not response or not response.strip():
        return valid_relations[-1] if valid_relations else "Other"

    # ---- 3. 剥离 Qwen3 的 think 块 ----
    response = response.strip()
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    response = re.sub(r'</think>', '', response).strip()

    # ---- 4. 判断是否双向（必须在所有使用之前）----
    is_directional = any('(' in r for r in valid_relations)

    # ---- 5. 从"答案:"等标记之后提取 ----
    answer_patterns = [
        r'答案[:：]\s*([A-Za-z\-_]+\(e[12],e[12]\))',
        r'最终答案[:：]\s*([A-Za-z\-_]+\(e[12],e[12]\))',
        r'final answer[:：]?\s*([A-Za-z\-_]+\(e[12],e[12]\))',
        r'your answer[:：]?\s*([A-Za-z\-_]+\(e[12],e[12]\))',
        r'(?:^|\n)\s*([A-Za-z\-_]+\(e[12],e[12]\))',
    ]
    for pat in answer_patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            candidate = m.group(1)
            for rel in valid_relations:
                if rel.lower() == candidate.lower():
                    return rel

    # ---- 6. 全文扫描，取位置最靠后的（模型最终答案通常在末尾）----
    found = []
    for rel in sorted(valid_relations, key=len, reverse=True):
        for m in re.finditer(re.escape(rel), response, re.IGNORECASE):
            found.append((m.start(), rel))
    if found:
        found.sort(key=lambda x: x[0], reverse=True)
        return found[0][1]

    # ---- 7. 双向模式下，尝试补方向 ----
    if is_directional:
        base_map = {}
        for r in valid_relations:
            base = r.split('(')[0]
            base_map.setdefault(base, []).append(r)
        for base in sorted(base_map.keys(), key=len, reverse=True):
            if base.lower() not in response.lower():
                continue
            if re.search(r'\(\s*e2\s*,\s*e1\s*\)', response, re.IGNORECASE):
                return f"{base}(e2,e1)"
            return f"{base}(e1,e2)"

    # ---- 8. 兜底 ----
    if "Other" in valid_relations:
        return "Other"
    return valid_relations[0]


# ========== 评估指标 ==========

def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """严格匹配（含方向）"""
    if len(predictions) != len(labels) or len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, label in zip(predictions, labels):
        if _normalize_label(pred) == _normalize_label(label):
            correct += 1
    return correct / len(predictions)


def compute_macro_f1(predictions: List[str], labels: List[str],
                     relation_names: List[str]) -> Dict[str, Any]:
    """Macro-averaged F1（SemEval-2010 官方指标）"""
    norm_space = [_normalize_label(r) for r in relation_names]
    tp = {r: 0 for r in norm_space}
    fp = {r: 0 for r in norm_space}
    fn = {r: 0 for r in norm_space}

    for pred, true in zip(predictions, labels):
        pred_n = _normalize_label(pred)
        true_n = _normalize_label(true)
        if pred_n == true_n and pred_n in tp:
            tp[pred_n] += 1
        else:
            if pred_n in fp:
                fp[pred_n] += 1
            if true_n in fn:
                fn[true_n] += 1

    f1s = {}
    for rel in relation_names:
        norm_rel = _normalize_label(rel)
        p = tp[norm_rel] / (tp[norm_rel] + fp[norm_rel]) if (tp[norm_rel] + fp[norm_rel]) > 0 else 0.0
        r = tp[norm_rel] / (tp[norm_rel] + fn[norm_rel]) if (tp[norm_rel] + fn[norm_rel]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s[rel] = f1

    macro_f1 = sum(f1s.values()) / len(f1s) if f1s else 0.0
    micro_acc = sum(tp.values()) / len(labels) if labels else 0.0

    return {
        "macro_f1": macro_f1,
        "accuracy": micro_acc,
        "per_relation": f1s,
        "support": {r: tp[_normalize_label(r)] + fn[_normalize_label(r)] for r in relation_names},
    }


def compute_f1(predictions: List[str], labels: List[str],
               relation_names: List[str]) -> Dict[str, Any]:
    """兼容旧接口"""
    return compute_macro_f1(predictions, labels, relation_names)