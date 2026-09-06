#!/usr/bin/env bash
# =============================================================================
# 防守方终端脚本 —— 图像痕迹隐匿与鲁棒检测基准（交付 Baseline：频域统计检测器）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_defense.sh
#
# 流程：拟合 FrequencyStatDefense → 干净集 CleanAUC/ACC/EER → 对固定攻击池 A0
#       （identity/jpeg/resize_recover/blur_sharpen/color_laundering/
#        frequency_laundering/composite_hidden）逐一计算 per-attack AUC →
#       池化 RobustAUC → 防御得分 (CleanAUC+RobustAUC)/2
# 输出：results/metrics.json
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_defense] 防守方 Baseline 评测（FrequencyStatDefense）"
"$PY" test.py --stage defense
