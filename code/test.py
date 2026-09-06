#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试代码 —— 图像痕迹隐匿与鲁棒检测基准（Benchmark）指标计算。

流程：
    攻击评测（run_attack.sh -> python test.py --stage attack）：
        1. 在训练集上拟合固定防御池 D0（frequency_stat + pixel_stat 两个检测器）；
        2. 在干净测试集上评测两检测器（CleanAUC / CleanACC，作为检测能力参照）；
        3. 对测试集全部 DeepFake 施加交付 Baseline 攻击（JPEG 重编码, quality=85）；
        4. 逐样本做合法性检查（形状/通道/dtype/NaN/大面积纯色）与视觉保真度量
           （SSIM / PSNR），计算两检测器上的攻击成功率 ASR 与平均 ASR。
        指标：CleanAUC / ASR / mean_ASR / mean_SSIM / mean_PSNR / valid_rate /
              qualified（资格线：valid_rate ≥ 0.8 且 mean_SSIM ≥ 0.92）

    防御评测（run_defense.sh -> python test.py --stage defense）：
        1. 在训练集上拟合交付 Baseline 检测器 FrequencyStatDefense；
        2. 干净测试集 CleanAUC / CleanACC / CleanEER；
        3. 对测试集 DeepFake 逐一施加固定攻击池 A0（identity / jpeg /
           resize_recover / blur_sharpen / color_laundering /
           frequency_laundering / composite_hidden），得到各攻击的 per-attack
           AUC（攻击后伪造图 + 原始真实图）与 RobustAUC（全部合法攻击后伪造图
           池化 + 原始真实图），composite_hidden 视为隐藏攻击 AUC。
        指标：CleanAUC / RobustAUC / per_attack_auc / hidden_attack_auc /
              防御得分 = (CleanAUC + RobustAUC) / 2

指标公式（与基准文档一致）：
    Accuracy(τ=0.5) = 正确预测数 / 样本总数
    AUC            = ROC 曲线下面积
    EER            = FPR == FNR 处的误报率
    ASR(a,d)       = (1/N_fake) · Σ_i 1[valid_i ∧ p_{f,i} < 0.5]
    SSIM / PSNR    = 原图与攻击后图的保真度量
    RobustAUC      = 池化全部合法攻击后伪造图 + 原始真实图的 AUC

用法：
    python test.py --stage attack      # 攻击方 Baseline 评测（默认）
    python test.py --stage defense     # 防守方 Baseline 评测
    python test.py --stage all         # 完整攻防评测
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from attack import build_attack_pool, JPEGReencodeAttack
from dataset import load_benchmark
from defense import (FIXED_DEFENSE_POOL, Defense, PixelStatDefense,
                     build_defense)

# 固定二分类阈值（赛题 §8：τ = 0.5）
TAU = 0.5
# 视觉保真软约束（赛题 §6.2：SSIM ≥ 0.92, PSNR ≥ 28dB）
SSIM_MIN = 0.92
PSNR_MIN = 28.0
# 攻击资格线（赛题 §11.1：合法攻击率 ≥ 0.80 且平均 SSIM ≥ 0.92）
VALID_RATE_MIN = 0.80


# --------------------------------------------------------------------------- #
# 统一评测协议：检测器训练
# --------------------------------------------------------------------------- #
def fit_detectors(pool_names):
    """在训练集上拟合固定防御池中的检测器，返回 {name: detector}。"""
    train, _ = load_benchmark()
    detectors = {}
    for name in pool_names:
        det = build_defense(name)
        t0 = time.time()
        det.fit(train["images"], train["labels"])
        print(f"[test] 检测器 {name} 拟合完成（{time.time()-t0:.1f}s）")
        detectors[name] = det
    return detectors, train


def scores_of(detector, images) -> np.ndarray:
    """批量推理 fake_probability。"""
    return np.array([detector.predict(im) for im in images], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 指标计算（严格对齐基准文档公式）
# --------------------------------------------------------------------------- #
def accuracy(labels: np.ndarray, scores: np.ndarray, tau: float = TAU) -> float:
    """Accuracy(τ) = 正确预测数 / 样本总数。"""
    pred = (scores >= tau).astype(int)
    return float(np.mean(pred == labels))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC；单类别时返回 0.5。"""
    if len(np.unique(labels)) < 2:
        return 0.5
    from sklearn import metrics as skm
    return float(skm.roc_auc_score(labels, scores))


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """等错误率 EER：FPR == FNR 处的误报率。"""
    if len(np.unique(labels)) < 2:
        return 0.0
    from sklearn import metrics as skm
    fpr, tpr, _ = skm.roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[idx])


def attack_success_rate(fake_scores: np.ndarray, valid_flags: np.ndarray) -> float:
    """ASR(a,d) = (1/N_fake)·Σ 1[valid ∧ p_f < τ]，值越大攻击越成功。"""
    if len(fake_scores) == 0:
        return 0.0
    succ = valid_flags & (fake_scores < TAU)
    return float(succ.mean())


# --------------------------------------------------------------------------- #
# 合法性检查与视觉保真（赛题 §6.1 / §6.2）
# --------------------------------------------------------------------------- #
def _flat_region_ratio(image: np.ndarray) -> float:
    """粗略估计大面积近纯色区域占比（8×8 网格块标准差 < 1 的比例）。"""
    img = np.asarray(image, dtype=np.int16)
    h, w, c = img.shape
    q = (img >> 4).astype(np.int16)
    code = q[..., 0] * 4096 + q[..., 1] * 64 + q[..., 2]
    gh, gw = 8, 8
    ys = np.linspace(0, h, gh, endpoint=False).astype(int)
    xs = np.linspace(0, w, gw, endpoint=False).astype(int)
    stds = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            y2 = ys[i + 1] if i + 1 < gh else h
            x2 = xs[j + 1] if j + 1 < gw else w
            block = code[y:y2, x:x2]
            if block.size:
                stds.append(float(block.std()))
    flat = sum(1 for s in stds if s < 1.0)
    return flat / max(len(stds), 1)


def check_legality(original: np.ndarray, attacked: np.ndarray) -> tuple:
    """逐样本合法性检查（赛题 §6.1）。

    返回 (valid: bool, reasons: list[str])。
    要求：宽高不变、通道数 3、可解析为 uint8、无 NaN/Inf、无大面积纯色区域。
    """
    reasons = []
    arr = np.asarray(attacked)
    ref = np.asarray(original)
    if arr.size == 0:
        return False, ["output is empty"]
    if arr.ndim != 3 or arr.shape[2] != 3:
        reasons.append(f"channel count changed: {arr.shape}")
    if arr.shape[0] != ref.shape[0] or arr.shape[1] != ref.shape[1]:
        reasons.append(f"size changed: {arr.shape[:2]} vs {ref.shape[:2]}")
    try:
        arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return False, ["pixels not parseable as uint8"]
    if not np.all(np.isfinite(arr.astype(np.float32))):
        reasons.append("contains NaN/Inf")
    if _flat_region_ratio(arr8) > 0.5:
        reasons.append(f"large flat region: {_flat_region_ratio(arr8):.2f}")
    return (len(reasons) == 0), reasons


def ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 11,
         sigma: float = 1.5, data_range: float = 255.0) -> float:
    """结构相似度 SSIM（逐通道高斯窗口，均值）；值越大越相似。"""
    from scipy import ndimage as ndi

    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ax = np.arange(win_size) - (win_size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mu1 = ndi.convolve(x, kernel, mode="reflect")
        mu2 = ndi.convolve(y, kernel, mode="reflect")
        s1 = ndi.convolve(x * x, kernel, mode="reflect") - mu1 * mu1
        s2 = ndi.convolve(y * y, kernel, mode="reflect") - mu2 * mu2
        s12 = ndi.convolve(x * y, kernel, mode="reflect") - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2)
        vals.append(float(np.mean(num / np.maximum(den, 1e-12))))
    return float(np.mean(vals))


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """峰值信噪比 PSNR（dB）；值越大越相似。"""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


# --------------------------------------------------------------------------- #
# 攻击评测
# --------------------------------------------------------------------------- #
def evaluate_attack(quality: int = 85, seed: int = 42, output_dir: str = "results"):
    train, test = load_benchmark()
    fake_mask = test["labels"] == 1
    real_mask = test["labels"] == 0
    fake_imgs = test["images"][fake_mask]
    real_imgs = test["images"][real_mask]
    fake_ids = [test["sample_ids"][i] for i in range(len(test["labels"])) if fake_mask[i]]
    n_fake = len(fake_imgs)

    # 1) 固定防御池 D0 训练 + 清洁能力参照
    pool_names = [n for n, _p in FIXED_DEFENSE_POOL]
    detectors, _ = fit_detectors(pool_names)

    clean = {}
    for name, det in detectors.items():
        s = scores_of(det, test["images"])
        clean[name] = {"auc": roc_auc(test["labels"], s),
                       "acc": accuracy(test["labels"], s)}

    # 2) 交付 Baseline 攻击：JPEG 重编码
    atk = JPEGReencodeAttack(quality=quality, seed=seed)
    t0 = time.time()
    attacked = np.stack([atk.apply(im) for im in fake_imgs])
    attack_runtime_ms = (time.time() - t0) * 1000.0 / max(n_fake, 1)

    # 3) 合法性 + 保真
    valid_flags = np.zeros(n_fake, dtype=bool)
    ssims, psnrs = np.zeros(n_fake), np.zeros(n_fake)
    invalid_reasons = {}
    for i in range(n_fake):
        ok, reasons = check_legality(fake_imgs[i], attacked[i])
        valid_flags[i] = ok
        if not ok:
            invalid_reasons[fake_ids[i]] = reasons
        ssims[i] = ssim(fake_imgs[i], attacked[i])
        psnrs[i] = psnr(fake_imgs[i], attacked[i])
    valid_rate = float(valid_flags.mean())
    mean_ssim = float(ssims.mean())
    mean_psnr = float(psnrs.mean())
    ssim_pass_rate = float((ssims >= SSIM_MIN).mean())
    psnr_pass_rate = float((psnrs >= PSNR_MIN).mean())
    qualified = bool(valid_rate >= VALID_RATE_MIN and mean_ssim >= SSIM_MIN)

    # 4) 各检测器上的 ASR 与 per-attack AUC
    asr_per_defense, auc_per_defense = {}, {}
    for name, det in detectors.items():
        fake_scores = scores_of(det, attacked)
        real_scores = scores_of(det, real_imgs)
        asr_per_defense[name] = attack_success_rate(fake_scores, valid_flags)
        lab = np.concatenate([np.ones(n_fake), np.zeros(len(real_imgs))])
        sc = np.concatenate([fake_scores, real_scores])
        auc_per_defense[name] = roc_auc(lab, sc)

    mean_asr = float(np.mean(list(asr_per_defense.values())))

    result = {
        "benchmark": "图像痕迹隐匿与鲁棒检测",
        "stage": "attack",
        "train_size": int(len(train["labels"])),
        "test_size": int(len(test["labels"])),
        "n_test_fake": n_fake,
        "n_test_real": int(real_mask.sum()),
        "seed": seed,
        "attack_baseline": "jpeg",
        "attack_params": {"quality": quality},
        "defense_pool": pool_names,
        "clean_auc": {k: round(v["auc"], 4) for k, v in clean.items()},
        "clean_acc": {k: round(v["acc"], 4) for k, v in clean.items()},
        "asr_per_defense": {k: round(v, 4) for k, v in asr_per_defense.items()},
        "auc_per_defense_after_attack": {k: round(v, 4) for k, v in auc_per_defense.items()},
        "mean_asr": round(mean_asr, 4),
        "valid_rate": round(valid_rate, 4),
        "mean_ssim": round(mean_ssim, 4),
        "mean_psnr": round(mean_psnr, 2),
        "ssim_pass_rate": round(ssim_pass_rate, 4),
        "psnr_pass_rate": round(psnr_pass_rate, 4),
        "qualified": qualified,
        "fidelity_thresholds": {"ssim_min": SSIM_MIN, "psnr_min": PSNR_MIN},
        "attack_runtime_ms_per_image": round(attack_runtime_ms, 3),
        "invalid_samples": invalid_reasons,
    }
    os.makedirs(output_dir, exist_ok=True)
    return result


# --------------------------------------------------------------------------- #
# 防御评测
# --------------------------------------------------------------------------- #
def evaluate_defense(output_dir: str = "results"):
    train, test = load_benchmark()
    fake_mask = test["labels"] == 1
    real_mask = test["labels"] == 0
    fake_imgs = test["images"][fake_mask]
    real_imgs = test["images"][real_mask]
    n_fake = len(fake_imgs)
    real_labels = np.zeros(int(real_mask.sum()), dtype=np.int64)

    # 1) 交付 Baseline 检测器
    det = Defense()
    t0 = time.time()
    det.fit(train["images"], train["labels"])
    print(f"[test] Defense 拟合完成（{time.time()-t0:.1f}s）")

    # 2) 清洁能力
    clean_scores = scores_of(det, test["images"])
    clean_auc = roc_auc(test["labels"], clean_scores)
    clean_acc = accuracy(test["labels"], clean_scores)
    clean_eer = eer(test["labels"], clean_scores)
    real_scores_clean = clean_scores[real_mask]

    # 3) 固定攻击池 A0 攻击后伪造图
    pool = build_attack_pool()
    robust_scores_all, robust_labels_all = [], []
    per_attack_auc, per_attack_asr = {}, {}
    n_legal_per_attack = {}
    for name, atk in pool:
        t0 = time.time()
        attacked = np.stack([atk.apply(im) for im in fake_imgs])
        atk_ms = (time.time() - t0) * 1000.0 / max(n_fake, 1)
        valid_flags = np.zeros(n_fake, dtype=bool)
        for i in range(n_fake):
            valid_flags[i] = check_legality(fake_imgs[i], attacked[i])[0]
        n_legal_per_attack[name] = int(valid_flags.sum())

        fake_scores = scores_of(det, attacked)
        # per-attack AUC：合法攻击后伪造图 + 原始真实图（非法样本记 p=1.0 视为被检出）
        pa_scores = np.where(valid_flags, fake_scores, 1.0)
        pa_lab = np.concatenate([np.ones(n_fake), real_labels])
        pa_sc = np.concatenate([pa_scores, real_scores_clean])
        per_attack_auc[name] = roc_auc(pa_lab, pa_sc)
        per_attack_asr[name] = attack_success_rate(fake_scores, valid_flags)
        print(f"[test] 攻击 {name:18s} 完成（{atk_ms:.1f} ms/图, "
              f"合法 {n_legal_per_attack[name]}/{n_fake}）")

        # Robust 池：合法攻击后伪造图
        legal_fake_scores = fake_scores[valid_flags]
        robust_scores_all.extend(list(legal_fake_scores))
        robust_labels_all.extend([1] * len(legal_fake_scores))

    # RobustAUC：池化全部合法攻击后伪造图 + 原始真实图
    robust_scores_all += list(real_scores_clean)
    robust_labels_all += [0] * len(real_scores_clean)
    robust_auc = roc_auc(np.asarray(robust_labels_all), np.asarray(robust_scores_all))
    robust_acc = accuracy(np.asarray(robust_labels_all), np.asarray(robust_scores_all))
    robust_eer = eer(np.asarray(robust_labels_all), np.asarray(robust_scores_all))

    hidden_attack_auc = per_attack_auc.get("composite_hidden")
    defense_score = (clean_auc + robust_auc) / 2.0

    result = {
        "benchmark": "图像痕迹隐匿与鲁棒检测",
        "stage": "defense",
        "train_size": int(len(train["labels"])),
        "test_size": int(len(test["labels"])),
        "n_test_fake": n_fake,
        "n_test_real": int(real_mask.sum()),
        "seed": 42,
        "defense_baseline": "frequency_stat",
        "clean_auc": round(clean_auc, 4),
        "clean_acc": round(clean_acc, 4),
        "clean_eer": round(clean_eer, 4),
        "robust_auc": round(robust_auc, 4),
        "robust_acc": round(robust_acc, 4),
        "robust_eer": round(robust_eer, 4),
        "per_attack_auc": {k: round(v, 4) for k, v in per_attack_auc.items()},
        "per_attack_asr": {k: round(v, 4) for k, v in per_attack_asr.items()},
        "n_legal_per_attack": n_legal_per_attack,
        "hidden_attack_auc": None if hidden_attack_auc is None else round(hidden_attack_auc, 4),
        "defense_score": round(defense_score, 4),
    }
    os.makedirs(output_dir, exist_ok=True)
    return result


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def print_attack_report(r: dict):
    print("\n" + "=" * 66)
    print("基准: 图像痕迹隐匿与鲁棒检测  |  阶段: 攻击评测（交付 Baseline: JPEG 重编码）")
    print("=" * 66)
    print(f"训练集 {r['train_size']} 张（真/假各半）   测试集 {r['test_size']} 张 "
          f"（真 {r['n_test_real']} / 假 {r['n_test_fake']}）")
    print(f"攻击参数: quality={r['attack_params']['quality']}  seed={r['seed']}")
    print("-" * 66)
    for name in r["defense_pool"]:
        print(f"检测器 {name:16s} CleanAUC={r['clean_auc'][name]:.4f}  "
              f"CleanACC={r['clean_acc'][name]:.4f}  "
              f"攻击后 ASR={r['asr_per_defense'][name]:.4f}  "
              f"攻击后 AUC={r['auc_per_defense_after_attack'][name]:.4f}")
    print(f"平均 ASR（跨检测器）      : {r['mean_asr']:.4f}")
    print("-" * 66)
    print(f"合法攻击率 valid_rate     : {r['valid_rate']:.4f}  (>= {VALID_RATE_MIN})")
    print(f"平均 SSIM                 : {r['mean_ssim']:.4f}  (>= {SSIM_MIN}, "
          f"通过率 {r['ssim_pass_rate']:.2f})")
    print(f"平均 PSNR                 : {r['mean_psnr']:.2f} dB  (>= {PSNR_MIN}, "
          f"通过率 {r['psnr_pass_rate']:.2f})")
    print(f"攻击资格线（valid>=0.8 且 SSIM>=0.92）: {'PASS' if r['qualified'] else 'FAIL'}")
    print(f"攻击耗时                   : {r['attack_runtime_ms_per_image']:.3f} ms/图")
    if r["invalid_samples"]:
        print(f"非法样本数: {len(r['invalid_samples'])}（详见 results/metrics.json）")
    print("=" * 66)


def print_defense_report(r: dict):
    print("\n" + "=" * 66)
    print("基准: 图像痕迹隐匿与鲁棒检测  |  阶段: 防御评测（交付 Baseline: FrequencyStat）")
    print("=" * 66)
    print(f"训练集 {r['train_size']} 张   测试集 {r['test_size']} 张")
    print("-" * 66)
    print(f"CleanAUC   = {r['clean_auc']:.4f}   CleanACC = {r['clean_acc']:.4f}   "
          f"CleanEER = {r['clean_eer']:.4f}")
    print(f"RobustAUC  = {r['robust_auc']:.4f}   RobustACC = {r['robust_acc']:.4f}   "
          f"RobustEER = {r['robust_eer']:.4f}")
    print(f"防御得分   = (CleanAUC + RobustAUC) / 2 = {r['defense_score']:.4f}")
    print("-" * 66)
    print("各攻击的 per-attack AUC（攻击后伪造图 + 原始真实图）与 ASR:")
    for name, auc in r["per_attack_auc"].items():
        print(f"  {name:18s} AUC={auc:.4f}   ASR={r['per_attack_asr'][name]:.4f}   "
              f"合法 {r['n_legal_per_attack'][name]}/{r['n_test_fake']}")
    print(f"隐藏组合攻击 AUC（composite_hidden）: {r['hidden_attack_auc']}")
    print("=" * 66)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="图像痕迹隐匿与鲁棒检测基准：Baseline 指标计算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["attack", "defense", "all"],
                        default="attack", help="评测阶段")
    parser.add_argument("--quality", type=int, default=85, help="JPEG 攻击质量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.stage in ("attack", "all"):
        res = evaluate_attack(quality=args.quality, seed=args.seed,
                              output_dir=os.path.dirname(args.output) or ".")
        print_attack_report(res)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n[test] 攻击指标已保存: {args.output}")

    if args.stage in ("defense", "all"):
        res = evaluate_defense(output_dir=os.path.dirname(args.output) or ".")
        print_defense_report(res)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n[test] 防御指标已保存: {args.output}")
