# # extract_validator_conf.py
# import re
# import json
# import os
#
#
# def main():
#     # ========================================================
#     # 日志路径配置
#     # 目前只开启 Qwen3 的日志提取。
#     # 等以后跑了 Qwen2.5，把下面 3 行 Qwen2.5 的注释删掉即可。
#     # ========================================================
#     log_files = {
#         # --- Qwen3-14B 日志 ---
#         "logs/full_semeval_qwen3-14b.log": "qwen3",
#         "logs/full_scierc_qwen3-14b.log": "qwen3",
#         "logs/full_wiki80_qwen3-14b.log": "qwen3",
#
#         # --- Qwen2.5-7B 日志（目前注释掉，后续开启） ---
#         # "logs/full_semeval_qwen2.5-7b.log": "qwen2.5",
#         # "logs/full_scierc_qwen2.5-7b.log": "qwen2.5",
#         # "logs/full_wiki80_qwen2.5-7b.log": "qwen2.5",
#     }
#
#     records = []
#     for lf, model_name in log_files.items():
#         if not os.path.exists(lf):
#             print(f"[跳过] 找不到日志: {lf}")
#             continue
#
#         print(f"[扫描] {lf} (模型: {model_name}) ...")
#         with open(lf, errors="ignore") as f:
#             for line in f:
#                 # 匹配格式: [Validator] #0: passed=True, conf=90
#                 m = re.search(r'\[Validator\]\s+#\d+:\s+passed=(True|False),\s+conf=(\d+)', line)
#                 if m:
#                     records.append({
#                         "model": model_name,
#                         "passed": m.group(1) == "True",
#                         "confidence": int(m.group(2))
#                     })
#
#     print(f"\n共提取到 {len(records)} 条 Validator 置信度记录")
#
#     if len(records) == 0:
#         print("❌ 未提取到任何数据。请确认日志文件中是否包含 '[Validator]' 的输出。")
#         return
#
#     # 简单统计一下当前提取到的数据
#     for m in set(r["model"] for r in records):
#         m_records = [r for r in records if r["model"] == m]
#         passed = [r["confidence"] for r in m_records if r["passed"]]
#         failed = [r["confidence"] for r in m_records if not r["passed"]]
#         print(f"--- {m.upper()} 统计 ---")
#         if passed: print(f"  passed=True : {len(passed)} 条, 平均置信度: {sum(passed) / len(passed):.1f}")
#         if failed: print(f"  passed=False: {len(failed)} 条, 平均置信度: {sum(failed) / len(failed):.1f}")
#
#     # 保存
#     os.makedirs("outputs", exist_ok=True)
#     out_path = "outputs/validator_conf_data.json"
#     with open(out_path, "w", encoding="utf-8") as fp:
#         json.dump(records, fp, indent=2)
#     print(f"\n✅ 数据已成功保存到: {out_path}")
#
#
# if __name__ == "__main__":
#     main()

# extract_validator_conf.py
import re
import json
import os


def main():
    # ========================================================
    # 日志配置
    #   - 合并日志：一个文件含多个数据集，脚本用 "=== XXX ===" 标记切分
    #   - 分开日志：value 直接给固定 dataset
    # 当前只开 Qwen3；等 Qwen2.5 日志就绪后取消注释即可。
    # ========================================================
    MERGED_LOGS = {
        # 文件路径 -> 默认 model（dataset 由内部标记决定）
        "logs/qwen3-14b_full_20261027_144757.log": "qwen3",
        # "logs/qwen2.5-7b_full_<TS>.log": "qwen2.5",
    }
    SPLIT_LOGS = {
        # 文件路径 -> (model, dataset)（用于已拆分好的日志）
        # "logs/full_semeval_qwen3-14b.log": ("qwen3", "semeval"),
    }

    # 合并日志里 section 标记 -> dataset 名
    SECTION_MAP = {
        "=== SemEval ===": "semeval",
        "=== SciERC ===":  "scierc",
        "=== Wiki80 ===":  "wiki80",
    }

    # [Validator] #0: passed=True, conf=90
    # 注意：[E+V Validator] 不匹配（正则要求 [ 后直接是 Validator），这正是我们想要的
    VALIDATOR_RE = re.compile(
        r'\[Validator\]\s+#\d+:\s+passed=(True|False),\s+conf=(\d+)'
    )
    SECTION_RE = re.compile(r'^\[[^\]]+\]\s+(=== (?:SemEval|SciERC|Wiki80) ===)')

    records = []

    # ===== 1) 合并日志：按 section 切分 =====
    for lf, model_name in MERGED_LOGS.items():
        if not os.path.exists(lf):
            print(f"[跳过] 找不到合并日志: {lf}")
            continue
        print(f"[扫描-合并] {lf} (模型: {model_name}) ...")
        current_dataset = None
        with open(lf, errors="ignore") as f:
            for line in f:
                m_sec = SECTION_RE.match(line)
                if m_sec:
                    current_dataset = SECTION_MAP.get(m_sec.group(1))
                    continue
                m = VALIDATOR_RE.search(line)
                if m:
                    if current_dataset is None:
                        continue  # section 标记之前的行，跳过
                    records.append({
                        "model": model_name,
                        "dataset": current_dataset,
                        "passed": m.group(1) == "True",
                        "confidence": int(m.group(2)),
                    })

    # ===== 2) 分开日志（备用） =====
    for lf, (model_name, dataset_name) in SPLIT_LOGS.items():
        if not os.path.exists(lf):
            print(f"[跳过] 找不到日志: {lf}")
            continue
        print(f"[扫描-单文件] {lf} (模型: {model_name}, 数据集: {dataset_name}) ...")
        with open(lf, errors="ignore") as f:
            for line in f:
                m = VALIDATOR_RE.search(line)
                if m:
                    records.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "passed": m.group(1) == "True",
                        "confidence": int(m.group(2)),
                    })

    print(f"\n共提取到 {len(records)} 条 Validator 置信度记录")

    if len(records) == 0:
        print("❌ 未提取到任何数据。")
        print("   请检查：")
        print("   1) 日志路径是否正确")
        print("   2) 日志里是否有 '=== SemEval ===' 等 section 标记")
        print("   3) 日志里是否有 '[Validator] #N: passed=..., conf=...' 行")
        return

    # ===== 3) 按 (model, dataset) 分组统计 + 数据量校验 =====
    EXPECTED = {  # 每模型每数据集的理论条数
        "semeval": 2717,
        "scierc":  974,
        "wiki80":  5600,
    }
    print("\n--- 分组统计 ---")
    for key in sorted(set((r["model"], r["dataset"]) for r in records)):
        m, ds = key
        m_records = [r for r in records
                     if r["model"] == m and r["dataset"] == ds]
        passed = [r["confidence"] for r in m_records if r["passed"]]
        failed = [r["confidence"] for r in m_records if not r["passed"]]
        print(f"[{m} / {ds}] 共 {len(m_records)} 条 "
              f"(期望 {EXPECTED.get(ds, '?')})")
        if passed:
            print(f"  passed=True : {len(passed):5d} 条, 均值 {sum(passed)/len(passed):.1f}")
        if failed:
            print(f"  passed=False: {len(failed):5d} 条, 均值 {sum(failed)/len(failed):.1f}")

    # ===== 4) 保存 =====
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/validator_conf_data.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)
    print(f"\n✅ 数据已保存到: {out_path}")
    print(f"   总条数: {len(records)}")


if __name__ == "__main__":
    main()