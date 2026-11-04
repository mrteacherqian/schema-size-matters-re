


# run_semeval.py
import os
from config import Config
from data_loader import SemEval2010Loader
from few_shot_examples import load_semeval_train, select_few_shot_examples
from baselines import PureLLM, FewShotPureLLM
from graph import RelationExtractionGraph
from graph_extractor_only import RelationExtractionGraphExtractorOnly
# from graph_no_validator import RelationExtractionGraphNoValidator
from graph_extractor_validator import RelationExtractionGraphExtractorValidator
from utils import (compute_accuracy, compute_macro_f1, ensure_dir, save_json,
                   SEMEVAL_19)


# vLLM 端点映射：model_key -> (base_url, served_model_name)
VLLM_ENDPOINTS = {
    "qwen3-14b":  ("http://localhost:8000/v1", "qwen3-14b"),
    "qwen2.5-7b": ("http://localhost:8001/v1", "qwen2.5-7b"),
    "qwen2.5-14b": ("http://localhost:8000/v1", "qwen2.5-14b"),
    "qwen3-8b":   ("http://localhost:8000/v1", "qwen3-8b"),
}


def run_semeval(model_key: str = "qwen3-14b",
                k_values=(1, 2, 3),
                max_test=None,
                provider: str = None):
    """
    Run complete experiments on SemEval-2010.
    Args:
        model_key: Model identifier (e.g., "qwen3-14b", "qwen2.5-7b").
        k_values: List of k values for Few-shot experiments.
        max_test: Maximum number of test samples (None for full dataset).
        provider: Override Config.MODEL_PROVIDER (e.g., "vllm", "local").
    """
    print("=" * 70)
    print(f"SemEval-2010 (19 类双向) | 模型: {model_key}")
    print("=" * 70)

    # ===== 1. 确定 provider =====
    if provider:
        Config.MODEL_PROVIDER = provider
    elif model_key in VLLM_ENDPOINTS:
        Config.MODEL_PROVIDER = "vllm"   # 默认 vllm

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

    # ===== 3. 加载数据 =====
    loader = SemEval2010Loader()
    test_samples = loader.parse_file(Config.SEMEVAL_TEST, max_samples=max_test, shuffle=True)
    test_inputs = loader.to_dict_list(test_samples)
    train_samples = load_semeval_train(Config.SEMEVAL_TRAIN, max_samples=None, shuffle=True)
    print(f"测试样本: {len(test_inputs)} | 训练样本: {len(train_samples)}")
    print(f"关系类别数: {len(SEMEVAL_19)}（19 类双向）")

    true_labels = [s["relation"] for s in test_inputs]
    test_sentence = test_inputs[0]["sentence"] if test_inputs else ""

    results = {
        "dataset": "SemEval-2010",
        "model": model_key,
        "provider": Config.MODEL_PROVIDER,
        "num_relations": len(SEMEVAL_19),
        "test_size": len(test_inputs),
    }

    # ===== 4. Zero-shot =====
    print("\n[1] Zero-shot")
    pure = PureLLM(relation_names=SEMEVAL_19)   # 用 Config.MODEL_PROVIDER
    preds = pure.predict_batch(test_inputs)
    acc = compute_accuracy(preds, true_labels)
    f1 = compute_macro_f1(preds, true_labels, SEMEVAL_19)
    results["zero_shot"] = {"accuracy": acc, "macro_f1": f1["macro_f1"]}
    print(f"  Acc={acc:.4f} | Macro-F1={f1['macro_f1']:.4f}")

    # ===== 5. Few-shot（4 策略 × k）=====
    print("\n[2] Few-shot (4 策略 × k)")
    results["few_shot"] = {}
    for strategy in ['random', 'similarity', 'hybrid', 'balanced']:
        results["few_shot"][strategy] = {}
        for k in k_values:
            examples = select_few_shot_examples(
                train_samples, test_sentence=test_sentence,
                k=k, strategy=strategy
            )
            model = FewShotPureLLM(
                examples=examples,
                relation_names=SEMEVAL_19,
            )
            preds = model.predict_batch(test_inputs)
            acc = compute_accuracy(preds, true_labels)
            f1 = compute_macro_f1(preds, true_labels, SEMEVAL_19)
            results["few_shot"][strategy][f"k={k}"] = {
                "accuracy": acc, "macro_f1": f1["macro_f1"]
            }
            print(f"  [{strategy}] k={k}: Acc={acc:.4f} | Macro-F1={f1['macro_f1']:.4f}")

    # ===== 6. Multi-Agent（完整版）=====
    print("\n[3] Full Multi-Agent")
    graph = RelationExtractionGraph(relation_names=SEMEVAL_19)
    r = graph.run(test_inputs, batch_size=10)
    results["multi_agent"] = r["metrics"]
    print(f"  Acc={r['metrics']['accuracy']:.4f} | Macro-F1={r['metrics']['macro_f1']:.4f}")

    # ===== 7. Ablations =====
    print("\n[4] Ablation: Extractor Only")
    g_ext = RelationExtractionGraphExtractorOnly(relation_names=SEMEVAL_19)
    r_ext = g_ext.run(test_inputs, batch_size=10)
    print(f"  Acc={r_ext['metrics']['accuracy']:.4f} | "
          f"Macro-F1={r_ext['metrics']['macro_f1']:.4f}")

    print("[4] Ablation: Extractor + Validator (no Refiner)")
    from graph_extractor_validator import RelationExtractionGraphExtractorValidator
    g_ev = RelationExtractionGraphExtractorValidator(relation_names=SEMEVAL_19)
    r_ev = g_ev.run(test_inputs, batch_size=10)
    print(f"  Acc={r_ev['metrics']['accuracy']:.4f} | "
          f"Macro-F1={r_ev['metrics']['macro_f1']:.4f}")

    results["ablation"] = {
        "extractor_only": r_ext["metrics"],
        "extractor_validator": r_ev["metrics"],
    }

    # ===== 8. 保存结果 =====
    suffix = "_pilot" if max_test is not None else ""
    ent_tag = "full"  # 新增：全量标记
    output_path = f"outputs/semeval_{model_key}_{Config.MODEL_PROVIDER}_results_4x3_1{suffix}_{ent_tag}.json"

    # output_path = f"outputs/semeval_{model_key}_{Config.MODEL_PROVIDER}_results431.json"
    save_json(results, output_path)
    print(f"\n结果已保存: {output_path}")
    return results



if __name__ == "__main__":
    import sys
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3-14b"

    # 支持命令行覆盖 provider
    provider_arg = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            provider_arg = sys.argv[idx + 1]

    run_semeval(model_key=model_key, provider=provider_arg)
