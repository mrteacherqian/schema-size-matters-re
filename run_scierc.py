# run_scierc.py
from config import Config
from data_loader import SciERCLoader
from few_shot_examples import select_few_shot_examples
from baselines import PureLLM, FewShotPureLLM
from graph import RelationExtractionGraph
from graph_extractor_only import RelationExtractionGraphExtractorOnly
# from graph_no_validator import RelationExtractionGraphNoValidator
from graph_extractor_validator import RelationExtractionGraphExtractorValidator
from utils import (compute_accuracy, compute_macro_f1, ensure_dir, save_json,
                   SCIERC_RELATIONS)


# vLLM 端点映射：model_key -> (base_url, served_model_name)
VLLM_ENDPOINTS = {
    "qwen3-14b":  ("http://localhost:8000/v1", "qwen3-14b"),
    "qwen2.5-7b": ("http://localhost:8001/v1", "qwen2.5-7b"),
    "qwen2.5-14b": ("http://localhost:8000/v1", "qwen2.5-14b"),
    "qwen3-8b":   ("http://localhost:8000/v1", "qwen3-8b"),
}


def run_scierc(model_key: str = "qwen3-14b",
               k_values=(1, 2, 3),
               max_test=None,
               provider: str = None):
    print("=" * 70)
    print(f"SciERC | 模型: {model_key}")
    print("=" * 70)

    # ===== 1. 确定 provider =====
    if provider:
        Config.MODEL_PROVIDER = provider
    elif model_key in VLLM_ENDPOINTS:
        Config.MODEL_PROVIDER = "vllm"

    # ===== 2. 根据 provider 和 model_key 配置端点 =====
    if Config.MODEL_PROVIDER == "vllm":
        if model_key in VLLM_ENDPOINTS:
            Config.VLLM_BASE_URL, Config.VLLM_MODEL_NAME = VLLM_ENDPOINTS[model_key]
        print(f"[Config] provider=vllm")
        print(f"[Config]   base_url  = {Config.VLLM_BASE_URL}")
        print(f"[Config]   model_name= {Config.VLLM_MODEL_NAME}")
    elif Config.MODEL_PROVIDER == "local":
        Config.set_model(model_key)
        print(f"[Config] provider=local")
        print(f"[Config]   model_path= {Config.LOCAL_MODEL_PATH}")
    elif Config.MODEL_PROVIDER == "deepseek":
        print(f"[Config] provider=deepseek")

    ensure_dir(Config.OUTPUT_DIR)

    # test_samples = SciERCLoader.parse_file(Config.SCIERC_TEST, max_samples=max_test)
    # train_samples = SciERCLoader.parse_file(Config.SCIERC_TRAIN, max_samples=2000)
    test_samples = SciERCLoader.parse_file(Config.SCIERC_TEST, max_samples=max_test, shuffle=True)
    train_samples = SciERCLoader.parse_file(Config.SCIERC_TRAIN, max_samples=None, shuffle=True)
    relation_names = SCIERC_RELATIONS
    print(f"测试样本: {len(test_samples)} | 训练样本: {len(train_samples)}")
    print(f"关系类别数: {len(relation_names)}")

    true_labels = [s["relation"] for s in test_samples]
    test_sentence = test_samples[0]["sentence"] if test_samples else ""

    results = {
        "dataset": "SciERC",
        "model": model_key,
        "provider": Config.MODEL_PROVIDER,
        "num_relations": len(relation_names),
        "test_size": len(test_samples),
    }

    # Zero-shot
    print("\n[1] Zero-shot")
    pure = PureLLM(relation_names=relation_names)
    preds = pure.predict_batch(test_samples)
    acc = compute_accuracy(preds, true_labels)
    f1 = compute_macro_f1(preds, true_labels, relation_names)
    results["zero_shot"] = {"accuracy": acc, "macro_f1": f1["macro_f1"]}
    print(f"  Acc={acc:.4f} | Macro-F1={f1['macro_f1']:.4f}")

    # Few-shot (4 strategies x k)
    print("\n[2] Few-shot (4 strategies x k)")
    results["few_shot"] = {}
    STRATEGIES = ['random', 'similarity', 'hybrid', 'balanced']
    for strategy in STRATEGIES:
        results["few_shot"][strategy] = {}
        for k in k_values:
            examples = select_few_shot_examples(
                train_samples, test_sentence=test_sentence,
                k=k, strategy=strategy
            )
            model = FewShotPureLLM(examples=examples, relation_names=relation_names)
            preds = model.predict_batch(test_samples)
            acc = compute_accuracy(preds, true_labels)
            f1 = compute_macro_f1(preds, true_labels, relation_names)
            results["few_shot"][strategy][f"k={k}"] = {
                "accuracy": acc, "macro_f1": f1["macro_f1"]
            }
            print(f"  [{strategy}] k={k}: Acc={acc:.4f} | Macro-F1={f1['macro_f1']:.4f}")

    # Multi-Agent
    print("\n[3] Full Multi-Agent")
    graph = RelationExtractionGraph(relation_names=relation_names)
    r = graph.run(test_samples, batch_size=10)
    results["multi_agent"] = r["metrics"]
    print(f"  Acc={r['metrics']['accuracy']:.4f} | Macro-F1={r['metrics']['macro_f1']:.4f}")

    # # Ablations
    # print("\n[4] Ablation: Extractor Only")
    # g_ext = RelationExtractionGraphExtractorOnly(relation_names=relation_names)
    # r_ext = g_ext.run(test_samples, batch_size=10)
    # print(f"  Acc={r_ext['metrics']['accuracy']:.4f} | "
    #       f"Macro-F1={r_ext['metrics']['macro_f1']:.4f}")
    #
    # print("[4] Ablation: Extractor + Refiner (no Validator)")
    # g_nv = RelationExtractionGraphNoValidator(relation_names=relation_names)
    # r_nv = g_nv.run(test_samples, batch_size=10)
    # print(f"  Acc={r_nv['metrics']['accuracy']:.4f} | "
    #       f"Macro-F1={r_nv['metrics']['macro_f1']:.4f}")
    #
    # results["ablation"] = {
    #     "extractor_only": r_ext["metrics"],
    #     "extractor_refiner": r_nv["metrics"],
    # }

    # Ablations
    print("\n[4] Ablation: Extractor Only")
    g_ext = RelationExtractionGraphExtractorOnly(relation_names=relation_names)
    r_ext = g_ext.run(test_samples, batch_size=10)
    print(f"  Acc={r_ext['metrics']['accuracy']:.4f} | "
          f"Macro-F1={r_ext['metrics']['macro_f1']:.4f}")

    print("[4] Ablation: Extractor + Validator (no Refiner)")
    from graph_extractor_validator import RelationExtractionGraphExtractorValidator
    g_ev = RelationExtractionGraphExtractorValidator(relation_names=relation_names)
    r_ev = g_ev.run(test_samples, batch_size=10)
    print(f"  Acc={r_ev['metrics']['accuracy']:.4f} | "
          f"Macro-F1={r_ev['metrics']['macro_f1']:.4f}")

    results["ablation"] = {
        "extractor_only": r_ext["metrics"],
        "extractor_validator": r_ev["metrics"],
    }
    # 如果 max_test 有值（比如200），存为 _pilot；否则存为全量名字
    suffix = "_pilot" if max_test is not None else ""
    ent_tag = "full"  # 新增：全量标记
    output_path = f"outputs/scierc_{model_key}_{Config.MODEL_PROVIDER}_results_4x3_1{suffix}_{ent_tag}.json"
    # output_path = f"outputs/scierc_{model_key}_{Config.MODEL_PROVIDER}_results431.json"  ##原始值
    save_json(results, output_path)
    print(f"\n结果已保存: {output_path}")
    return results


if __name__ == "__main__":
    import sys
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3-14b"

    provider_arg = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            provider_arg = sys.argv[idx + 1]

    run_scierc(model_key=model_key, provider=provider_arg)