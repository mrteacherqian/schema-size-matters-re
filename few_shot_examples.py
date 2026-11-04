




# few_shot_examples.py - 完整版（支持相似度+多样性选择）
import random
import re
from typing import List, Dict, Optional
import numpy as np

ENCODER_MODEL_PATH = "./model/all-MiniLM-L6-v2"

_encoder = None

_EMB_CACHE = {}

def _encode_corpus_cached(encoder, texts, cache_key):
    if cache_key in _EMB_CACHE:
        return _EMB_CACHE[cache_key]
    embs = _encode_corpus(encoder, texts)
    _EMB_CACHE[cache_key] = embs
    return embs

def get_encoder(model_name: str = None):
    global _encoder
    if _encoder is not None:
        return _encoder
    if model_name is None:
        model_name = ENCODER_MODEL_PATH
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[Encoder] 加载: {model_name}")
        try:
            _encoder = SentenceTransformer(model_name, device='cuda')
        except Exception:
            _encoder = SentenceTransformer(model_name, device='cpu')
        # print(f"[Encoder] 加载成功, dim={_encoder.get_sentence_embedding_dimension()}")  ###未来警告
        print(f"[Encoder] 加载成功, dim={_encoder.get_embedding_dimension()}")

    except ImportError:
        print("[Encoder] sentence-transformers 未安装")
        _encoder = None
    except Exception as e:
        print(f"[Encoder] 加载失败: {e}")
        _encoder = None
    return _encoder


def _encode_corpus(encoder, texts: List[str]) -> np.ndarray:
    embs = encoder.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _encode_query(encoder, text: str) -> np.ndarray:
    emb = encoder.encode([text], convert_to_numpy=True)[0]
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def select_random(train_samples: List[Dict], k: int = 3, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    if len(train_samples) <= k:
        return train_samples.copy()
    shuffled = train_samples.copy()
    random.shuffle(shuffled)
    return shuffled[:k]


def select_similarity(train_samples: List[Dict], test_sentence: str, k: int = 3) -> List[Dict]:
    if len(train_samples) <= k:
        return train_samples.copy()
    encoder = get_encoder()
    if encoder is None:
        return select_random(train_samples, k)

    # embs = _encode_corpus(encoder, [s.get('sentence', '') for s in train_samples])
    cache_key = (id(train_samples), len(train_samples), id(encoder))
    embs = _encode_corpus_cached(encoder, [s.get('sentence', '') for s in train_samples], cache_key)
    q = _encode_query(encoder, test_sentence)
    sims = np.dot(embs, q)
    top = np.argsort(sims)[-k:][::-1]
    return [train_samples[i] for i in top]


def select_hybrid(train_samples: List[Dict], test_sentence: str, k: int = 3,
                  top_n: int = 20, diversity_weight: float = 0.3) -> List[Dict]:
    if len(train_samples) <= k:
        return train_samples.copy()
    encoder = get_encoder()
    if encoder is None:
        return select_random(train_samples, k)

    top_n = min(top_n, len(train_samples))
    # embs = _encode_corpus(encoder, [s.get('sentence', '') for s in train_samples])
    cache_key = (id(train_samples), len(train_samples), id(encoder))
    embs = _encode_corpus_cached(encoder, [s.get('sentence', '') for s in train_samples], cache_key)
    q = _encode_query(encoder, test_sentence)
    sims = np.dot(embs, q)

    top_idx = np.argsort(sims)[-top_n:][::-1]
    cand_samples = [train_samples[i] for i in top_idx]
    cand_embs = embs[top_idx]

    selected = []
    remaining = list(range(len(cand_samples)))

    for _ in range(k):
        best_score = -float('inf')
        best_j = -1
        for j in remaining:
            sim_score = float(sims[top_idx[j]])
            if selected:
                sel_embs = cand_embs[selected]
                dists = 1 - np.dot(sel_embs, cand_embs[j])
                div_score = float(np.mean(dists))
            else:
                div_score = 0.5
            score = (1 - diversity_weight) * sim_score + diversity_weight * div_score
            if score > best_score:
                best_score = score
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)

    return [cand_samples[i] for i in selected]


def select_balanced(train_samples: List[Dict], test_sentence: str, k: int = 3) -> List[Dict]:
    if len(train_samples) <= k:
        return train_samples.copy()

    groups = {}
    for s in train_samples:
        rel = s.get('relation', 'Other')
        # rel_name = re.sub(r'\(.*?\)', '', rel).strip()
        rel_name = rel.strip()
        groups.setdefault(rel_name, []).append(s)

    encoder = get_encoder()
    selected = []
    rel_names = list(groups.keys())
    random.seed(42)
    random.shuffle(rel_names)

    for rel in rel_names:
        if len(selected) >= k:
            break
        grp = groups[rel]
        if len(grp) == 1:
            selected.append(grp[0])
        else:
            selected.extend(select_similarity(grp, test_sentence, k=1))

    if len(selected) < k:
        sel_ids = {id(x) for x in selected}
        remaining = [s for s in train_samples if id(s) not in sel_ids]
        if remaining:
            selected.extend(select_similarity(remaining, test_sentence, k=k - len(selected)))
    return selected[:k]


def select_few_shot_examples(train_samples: List[Dict],
                             test_sentence: str = None,
                             k: int = 3, seed: int = 42,
                             strategy: str = 'hybrid') -> List[Dict]:
    if strategy == 'random':
        return select_random(train_samples, k, seed)
    if strategy == 'similarity':
        if test_sentence is None:
            return select_random(train_samples, k, seed)
        return select_similarity(train_samples, test_sentence, k)
    if strategy == 'balanced':
        if test_sentence is None:
            return select_random(train_samples, k, seed)
        return select_balanced(train_samples, test_sentence, k)
    # hybrid (default)
    if test_sentence is None:
        return select_random(train_samples, k, seed)
    return select_hybrid(train_samples, test_sentence, k)


# def load_semeval_train(train_file_path: str, max_samples: int = None) -> List[Dict]:
#     from data_loader import SemEval2010Loader
#     loader = SemEval2010Loader()
#     samples = loader.parse_file(train_file_path, max_samples=max_samples)
#     return loader.to_dict_list(samples)

def load_semeval_train(train_file_path: str, max_samples: int = None,
                       shuffle: bool = False, seed: int = 42) -> List[Dict]:
    from data_loader import SemEval2010Loader
    loader = SemEval2010Loader()
    samples = loader.parse_file(
        train_file_path,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
    )
    return loader.to_dict_list(samples)

def save_examples_to_file(examples: List[Dict], file_path: str):
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def load_examples_from_file(file_path: str) -> List[Dict]:
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)