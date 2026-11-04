# data_loader.py
import json
import re
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from utils import extract_entities_from_text


@dataclass
class RelationSample:
    id: int
    sentence: str
    e1: str
    e2: str
    relation: str
    comment: str = ""
    sentence_with_tags: str = ""


# ============================================================
# SemEval-2010 Task 8
# ============================================================
class SemEval2010Loader:
    """SemEval-2010 Task 8 加载器（19 类双向）"""

    RELATION_MAP = {
        'Cause-Effect(e1,e2)': 0, 'Cause-Effect(e2,e1)': 1,
        'Component-Whole(e1,e2)': 2, 'Component-Whole(e2,e1)': 3,
        'Content-Container(e1,e2)': 4, 'Content-Container(e2,e1)': 5,
        'Entity-Destination(e1,e2)': 6, 'Entity-Destination(e2,e1)': 7,
        'Entity-Origin(e1,e2)': 8, 'Entity-Origin(e2,e1)': 9,
        'Instrument-Agency(e1,e2)': 10, 'Instrument-Agency(e2,e1)': 11,
        'Member-Collection(e1,e2)': 12, 'Member-Collection(e2,e1)': 13,
        'Message-Topic(e1,e2)': 14, 'Message-Topic(e2,e1)': 15,
        'Product-Producer(e1,e2)': 16, 'Product-Producer(e2,e1)': 17,
        'Other': 18,
    }
    IDX_TO_RELATION = {v: k for k, v in RELATION_MAP.items()}

    @classmethod
    def parse_file(cls, file_path: str, max_samples: int = None,
                   shuffle: bool = False, seed: int = 42) -> List[RelationSample]:
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            match = re.match(r'(\d+)\s+"(.*)"', line)
            if not match:
                i += 1
                continue

            sample_id = int(match.group(1))
            sentence_with_tags = match.group(2)
            clean_sentence, e1, e2 = extract_entities_from_text(sentence_with_tags)

            i += 1
            if i >= len(lines):
                break
            relation_line = lines[i].strip()

            i += 1
            comment = ""
            if i < len(lines) and lines[i].strip().startswith('Comment:'):
                comment = lines[i].strip()
                i += 1

            samples.append(RelationSample(
                id=sample_id, sentence=clean_sentence,
                e1=e1, e2=e2, relation=relation_line,
                comment=comment, sentence_with_tags=sentence_with_tags
            ))

        # 先打乱再截断
        if shuffle:
            random.Random(seed).shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    @classmethod
    def to_dict_list(cls, samples: List[RelationSample]) -> List[Dict]:
        return [
            {
                "id": s.id,
                "sentence": s.sentence,
                "e1": s.e1,
                "e2": s.e2,
                "relation": s.relation,
                "relation_id": cls.RELATION_MAP.get(s.relation, 18),
                "comment": s.comment,
            }
            for s in samples
        ]


# ============================================================
# Wiki80（OpenNRE 格式）
# ============================================================
class Wiki80Loader:
    """
    Wiki80 加载器。

    数据格式（每行一个 JSON）：
      {
        "token": [...],                              # 可能是 token 或 tokens
        "h": {"name": "...", "id": "...", "pos": [start, end]},   # 或 "h": "字符串"
        "t": {...},
        "relation": "..."
      }

    注意：
      - 字段名可能是 "token"（单数，TACRED 系）或 "tokens"（复数）
      - pos 是 [start, end) 左闭右开（exclusive），对应 Python 切片 tokens[start:end]
    """

    @classmethod
    def parse_file(cls, file_path: str, max_samples: int = None,
                   shuffle: bool = False, seed: int = 42) -> List[Dict]:
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = cls._parse_item(item, i)
                if parsed:
                    samples.append(parsed)

        # 关键：先打乱再截断，避免取到同一关系类的连续样本
        if shuffle:
            random.Random(seed).shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    @classmethod
    def _parse_item(cls, item: Dict, idx: int) -> Optional[Dict]:
        # 兼容 token / tokens 两种字段名
        tokens = item.get("tokens") or item.get("token") or []
        if not tokens:
            sent = item.get("sentence", "")
            tokens = sent.split() if sent else []

        h_field = item.get("h")
        t_field = item.get("t")
        relation = item.get("relation", "")

        # ===== 格式B：h/t 是 {"name":..., "id":..., "pos":[s,e]}（exclusive）=====
        if isinstance(h_field, dict) and "pos" in h_field:
            h_start, h_end = h_field["pos"]
            t_start, t_end = t_field["pos"]

            # pos 是 exclusive，用 tokens[start:end] 取实体文本
            h_name = h_field.get("name") or " ".join(tokens[h_start:h_end])
            t_name = t_field.get("name") or " ".join(tokens[t_start:t_end])

            sentence = cls._insert_markers(tokens, h_start, h_end, t_start, t_end)
            return {
                "id": idx,
                "sentence": sentence,
                "e1": h_name,
                "e2": t_name,
                "relation": relation,
            }

        # ===== 格式A：h/t 是字符串，tokens 里带 <h></h><t></t> 标记 =====
        if isinstance(h_field, str):
            sentence, e1, e2 = cls._extract_from_marked_tokens(tokens)
            return {
                "id": idx,
                "sentence": sentence,
                "e1": e1 or h_field,
                "e2": e2 or (t_field if isinstance(t_field, str) else ""),
                "relation": relation,
            }

        return None

    @staticmethod
    def _extract_from_marked_tokens(tokens):
        text = " ".join(tokens)
        text = text.replace("<h>", "<e1>").replace("</h>", "</e1>")
        text = text.replace("<t>", "<e2>").replace("</t>", "</e2>")
        e1_m = re.search(r"<e1>(.*?)</e1>", text)
        e2_m = re.search(r"<e2>(.*?)</e2>", text)
        e1 = e1_m.group(1).strip() if e1_m else ""
        e2 = e2_m.group(1).strip() if e2_m else ""
        clean = re.sub(r"</?e1>|</?e2>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean, e1, e2

    @staticmethod
    def _insert_markers(tokens, h_start, h_end, t_start, t_end):
        """
        在 token 序列中插入 <e1></e1> 和 <e2></e2> 标记。
        注意：h_end / t_end 是 exclusive 上界（tokens[start:end] 是实体）。
        """
        parts = []
        for i, tok in enumerate(tokens):
            if i == h_start:
                parts.append("<e1>")
            if i == t_start:
                parts.append("<e2>")
            parts.append(tok)
            if i == h_end - 1:
                parts.append("</e1>")
            if i == t_end - 1:
                parts.append("</e2>")
        text = " ".join(parts)
        text = re.sub(r"<e1>\s+", "<e1>", text)
        text = re.sub(r"\s+</e1>", "</e1>", text)
        text = re.sub(r"<e2>\s+", "<e2>", text)
        text = re.sub(r"\s+</e2>", "</e2>", text)
        return re.sub(r"</?e1>|</?e2>", "", text).strip()

    @classmethod
    def get_relation_names(cls, file_path: str) -> List[str]:
        rels = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    r = item.get("relation", "")
                    if r:
                        rels.add(r)
                except json.JSONDecodeError:
                    continue
        return sorted(rels)


# ============================================================
# SciERC（JSONL，每行一个文档）
# ============================================================
# class SciERCLoader:
#     """
#     SciERC 加载器。
#
#     数据格式：每行一个文档的 JSON
#       {
#         "doc_key": "...",
#         "sentences": [[tok, tok, ...], ...],
#         "ner":  [[[start, end, type], ...], ...],
#         "relations": [[[h_start, h_end, t_start, t_end, rel], ...], ...]
#       }
#
#     注意：SciERC 的 pos 是 inclusive（左闭右闭），[start, end] 表示 tokens[start:end+1]
#     """
#
#     @classmethod
#     def parse_file(cls, file_path: str, max_samples: int = None,
#                    shuffle: bool = False, seed: int = 42) -> List[Dict]:
#         samples = []
#         with open(file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     doc = json.loads(line)
#                 except json.JSONDecodeError:
#                     continue
#
#                 doc_key = doc.get("doc_key", "")
#                 sentences = doc.get("sentences", [])
#                 all_ners = doc.get("ner", [])
#                 all_rels = doc.get("relations", [])
#
#                 # 兼容 ner/relations 是展平的情况
#                 if all_ners and isinstance(all_ners[0], list) \
#                         and all_ners[0] and isinstance(all_ners[0][0], int):
#                     all_ners = [all_ners]
#                 if all_rels and isinstance(all_rels[0], list) \
#                         and all_rels[0] and isinstance(all_rels[0][0], int):
#                     all_rels = [all_rels]
#
#                 for sent_idx, tokens in enumerate(sentences):
#                     sent_rels = all_rels[sent_idx] if sent_idx < len(all_rels) else []
#                     for rel in sent_rels:
#                         if len(rel) < 5:
#                             continue
#                         h_start, h_end, t_start, t_end, rel_type = rel[:5]
#                         if h_start < 0 or t_start < 0:
#                             continue
#                         # SciERC 的 pos 是 inclusive
#                         h_text = " ".join(tokens[h_start:h_end + 1])
#                         t_text = " ".join(tokens[t_start:t_end + 1])
#                         sentence = cls._insert_markers(
#                             tokens, h_start, h_end, t_start, t_end
#                         )
#                         samples.append({
#                             "id": f"{doc_key}_{sent_idx}",
#                             "sentence": sentence,
#                             "e1": h_text,
#                             "e2": t_text,
#                             "relation": rel_type,
#                         })
#
#         # 先打乱再截断
#         if shuffle:
#             random.Random(seed).shuffle(samples)
#         if max_samples is not None:
#             samples = samples[:max_samples]
#         return samples
#
#     @staticmethod
#     def _insert_markers(tokens, h_start, h_end, t_start, t_end):
#         """
#         SciERC 的 pos 是 inclusive（[start, end] 表示 tokens[start:end+1]），
#         所以标记加在 i == h_end 之后。
#         """
#         parts = []
#         for i, tok in enumerate(tokens):
#             if i == h_start:
#                 parts.append("<e1>")
#             if i == t_start:
#                 parts.append("<e2>")
#             parts.append(tok)
#             if i == h_end:
#                 parts.append("</e1>")
#             if i == t_end:
#                 parts.append("</e2>")
#         text = " ".join(parts)
#         text = re.sub(r"<e1>\s+", "<e1>", text)
#         text = re.sub(r"\s+</e1>", "</e1>", text)
#         text = re.sub(r"<e2>\s+", "<e2>", text)
#         text = re.sub(r"\s+</e2>", "</e2>", text)
#         return re.sub(r"</?e1>|</?e2>", "", text).strip()

class SciERCLoader:
    """
    SciERC 加载器。

    数据格式（每行一个 JSON）：
      - 顶层可能是单个文档 dict，也可能是文档列表 [doc1, doc2, ...]
      - 文档结构：
        {
          "tokens": ["...", ...],
          "entities": [{"type": "Task", "start": 0, "end": 4}, ...],
          "relations": [{"type": "Part-of", "head": 0, "tail": 3}, ...],
          "orig_id": "..."
        }
      - entities 的 start/end 是 token 索引（inclusive，end 是最后一个 token 的下标）
      - relations 的 head/tail 是 entities 列表的下标
    """

    @classmethod
    def parse_file(cls, file_path: str, max_samples: int = None,
                   shuffle: bool = False, seed: int = 42) -> List[Dict]:
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 兼容 dict 和 list 两种顶层结构
                if isinstance(parsed, dict):
                    docs = [parsed]
                elif isinstance(parsed, list):
                    docs = parsed
                else:
                    continue

                for doc in docs:
                    if isinstance(doc, dict):
                        samples.extend(cls._parse_doc(doc))

        if shuffle:
            random.Random(seed).shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    @classmethod
    def _parse_doc(cls, doc: Dict) -> List[Dict]:
        out = []
        tokens = doc.get("tokens", [])
        entities = doc.get("entities", [])
        relations = doc.get("relations", [])
        doc_id = doc.get("orig_id", doc.get("doc_key", ""))

        if not tokens or not entities or not relations:
            return out

        for rel_idx, rel in enumerate(relations):
            if not isinstance(rel, dict):
                continue

            rel_type = rel.get("type", "")
            head_idx = rel.get("head")
            tail_idx = rel.get("tail")

            # head/tail 是 entities 列表的下标，不是 token 下标
            if head_idx is None or tail_idx is None:
                continue
            if not (0 <= head_idx < len(entities)):
                continue
            if not (0 <= tail_idx < len(entities)):
                continue

            h_ent = entities[head_idx]
            t_ent = entities[tail_idx]

            h_start = h_ent.get("start")
            h_end = h_ent.get("end")
            t_start = t_ent.get("start")
            t_end = t_ent.get("end")

            if None in (h_start, h_end, t_start, t_end):
                continue

            # SciERC 的 start/end 是 exclusive（半开区间 [start, end)），tokens[start:end] 即实体文本
            h_text = " ".join(tokens[h_start:h_end])
            t_text = " ".join(tokens[t_start:t_end])
            sentence = cls._insert_markers(tokens, h_start, h_end, t_start, t_end)

            out.append({
                "id": f"{doc_id}_{rel_idx}",
                "sentence": sentence,
                "e1": h_text,
                "e2": t_text,
                "relation": rel_type,
            })
        return out

    @staticmethod
    def _insert_markers(tokens, h_start, h_end, t_start, t_end):
        """SciERC 的 pos 是 exclusive（半开区间 [start, end)）"""
        parts = []
        for i, tok in enumerate(tokens):
            if i == h_start:
                parts.append("<e1>")
            if i == t_start:
                parts.append("<e2>")
            parts.append(tok)

            if i == h_end - 1:  # end 是 exclusive 上界，最后一个 token 的下标是 end-1
                parts.append("</e1>")
            if i == t_end - 1:
                parts.append("</e2>")
        text = " ".join(parts)
        text = re.sub(r"<e1>\s+", "<e1>", text)
        text = re.sub(r"\s+</e1>", "</e1>", text)
        text = re.sub(r"<e2>\s+", "<e2>", text)
        text = re.sub(r"\s+</e2>", "</e2>", text)
        return re.sub(r"</?e1>|</?e2>", "", text).strip()