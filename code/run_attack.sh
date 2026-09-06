#!/usr/bin/env bash
# =============================================================================
# 攻击方终端脚本 —— 图像痕迹隐匿与鲁棒检测基准（交付 Baseline：JPEG 重编码）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_attack.sh
#
# 流程：拟合固定防御池 D0（frequency_stat / pixel_stat）→ 干净集 CleanAUC 参照
#       → 对测试集 DeepFake 施加 JPEG(quality=85) 隐匿 → 合法性/SSIM/PSNR
#       → 各检测器 ASR 与平均 ASR → 资格线判定
# 输出：results/metrics.json
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_attack] 攻击方 Baseline 评测（JPEG 重编码, quality=85）"
"$PY" test.py --stage attack
