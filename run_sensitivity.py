# run_sensitivity.py
"""敏感性分析：200 条分层子集，conf_threshold=60 强制触发 Refiner"""
import json
from config import Config
from data_loader import SemEval2010Loader, SciERCLoader, Wiki80Loader
from graph import RelationExtractionGraph
from graph_extractor_validator import RelationExtractionGraphExtractorValidator
from utils import compute_accuracy, save_json, SEMEVAL_19, SCIERC_RELATIONS


def run_one(dataset, model_key, max_test=200, conf_threshold=60):
    if model_key in ("qwen3-14b", "qwen2.5-7b"):
        Config.MODEL_PROVIDER = "vllm"
        Config.VLLM_BASE_URL, Config.VLLM_MODEL_NAME = {
            "qwen3-14b":  ("http://localhost:8000/v1", "qwen3-14b"),
            "qwen2.5-7b": ("http://localhost:8001/v1", "qwen2.5-7b"),
        }[model_key]

    if dataset == "semeval":
        loader = SemEval2010Loader()
        test = loader.to_dict_list(loader.parse_file(Config.SEMEVAL_TEST,
                                                     max_samples=max_test, shuffle=True))
        rels = SEMEVAL_19
    elif dataset == "scierc":
        test = SciERCLoader.parse_file(Config.SCIERC_TEST, max_samples=max_test, shuffle=True)
        rels = SCIERC_RELATIONS
    elif dataset == "wiki80":
        test = Wiki80Loader.parse_file(Config.WIKI80_TEST, max_samples=max_test, shuffle=True)
        rels = Wiki80Loader.get_relation_names(Config.WIKI80_TRAIN)

    labels = [s["relation"] for s in test]

    # 基线：Extractor Only
    from graph_extractor_only import RelationExtractionGraphExtractorOnly
    g_ext = RelationExtractionGraphExtractorOnly(relation_names=rels)
    r_ext = g_ext.run(test, batch_size=10)

    # 强制触发：conf_threshold=60
    g_ma = RelationExtractionGraph(relation_names=rels, conf_threshold=conf_threshold)
    r_ma = g_ma.run(test, batch_size=10)

    out = {
        "dataset": dataset, "model": model_key, "N": len(test),
        "conf_threshold": conf_threshold,
        "extractor_only": r_ext["metrics"],
        "full_ma_forced": r_ma["metrics"],
    }
    save_json(out, f"outputs/sensitivity_{dataset}_{model_key}_conf{conf_threshold}.json")
    print(f"[Sensitivity|{dataset}|{model_key}] "
          f"ExtOnly={r_ext['metrics']['accuracy']:.4f} → "
          f"MA(conf={conf_threshold})={r_ma['metrics']['accuracy']:.4f} | "
          f"triggered={r_ma['metrics']['refiner_triggered']}/{len(test)}")
    return out



if __name__ == "__main__":
    import sys
    models = sys.argv[2:] if len(sys.argv) > 2 and sys.argv[1] == "--models" \
             else ["qwen3-14b", "qwen2.5-7b"]
    for ds in ("scierc", "wiki80", "semeval"):
        for mk in models:
            run_one(ds, mk)