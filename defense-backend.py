#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御代码 —— 图像痕迹隐匿与鲁棒检测基准（Baseline：频域统计检测器）。

赛题中，防御方需提交 DeepFake 检测算法：仅根据最终输入图像（可能已被后处理
攻击隐匿）输出其为 DeepFake 的概率 fake_probability ∈ [0, 1]，同时应对未处理
DeepFake、公开后处理攻击与隐藏组合攻击，并保持真实图像低误报。

真实世界中的 DeepFake 检测器（见 DeepfakeBench）多为 CNN（Xception、MesoNet、
EfficientNet 等）或频域特征检测器（F3Net、SPSL、SRM 等），输入统一为
256×256 人脸裁剪、归一化到 [-1,1] 后送入网络。本基准在 CPU 上交付一个
**轻量化实现**：对赛题材料 deepguard_backend 中 frequency_stat 检测器（SRM
噪声残差 + 分块 DCT + FFT 频带统计 + 颜色统计特征 + 逻辑回归）的忠实复刻，
作为交付 Baseline **FrequencyStatDefense**（对应防御固定池 D0-3 频域特征家族）。

防御固定池 D0 另含一个空域/像素统计检测器 PixelStatDefense（通道统计 + 梯度
统计 + 逻辑回归），用于衡量攻击的跨检测器迁移能力。

接口：
    defend(request) -> {"fake_probability": float}   # 赛题 §7.4 提交接口

用法：
    python defense.py                  # 在训练集上拟合检测器，输出清洁集自检结果
    python defense.py --detector pixel  # 指定像素统计检测器
"""


import argparse
import os

import numpy as np

from dataset import load_benchmark

# --------------------------------------------------------------------------- #
# 频域/噪声残差特征（忠实于材料 deepguard_backend/defenses/frequency_stat.py）
# --------------------------------------------------------------------------- #
# SRM 高通噪声残差滤波器（标准 17 个中的 3 个）
_SRM_FILTERS = [
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
]


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II（基于 scipy.fft.dct，逐轴正交归一）。"""
    from scipy.fft import dct
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def extract_frequency_features(image: np.ndarray) -> np.ndarray:
    """紧凑频域 + 噪声残差特征向量（31 维）。

    组成：
    1. 3 个 SRM 高通残差滤波器的均值/标准差/绝对均值/99%-1% 分位差（12 维）；
    2. 8×8 分块 DCT 能量谱的 5 个代表性系数（低频 0,1,9 与高频 56,63）的
       log1p 能量与高/低频能量比（6 维）；
    3. 全局 FFT 幅度谱低/高频带（半径 <15% / >40%）的均值与标准差（4 维）；
    4. 每通道颜色统计：均值/标准差/极差（9 维）。
    """
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gray = np.mean(img, axis=2)

    feats: list = []
    # --- SRM 噪声残差统计 ---
    for kern in _SRM_FILTERS:
        res = ndi.convolve(gray, kern, mode="reflect")
        feats += [float(res.mean()), float(res.std()),
                  float(np.mean(np.abs(res))),
                  float(np.percentile(res, 99) - np.percentile(res, 1))]
    # --- 8×8 分块 DCT 能量 ---
    h, w = gray.shape
    bh, bw = 8, 8
    acc = np.zeros((bh, bw), dtype=np.float32)
    count = 0
    for i in range(0, h - bh + 1, bh):
        for j in range(0, w - bw + 1, bw):
            acc += np.abs(_dct2(gray[i:i + bh, j:j + bw]))
            count += 1
    if count > 0:
        acc /= count
    flat = acc.flatten()
    feats += [float(np.log1p(flat[k])) for k in (0, 1, 9, 56, 63)]
    low = float(np.sum(acc[:2, :2]))
    high = float(np.sum(acc[4:, 4:]))
    feats += [float(np.log1p(high / (low + 1e-6)))]
    # --- 全局 FFT 幅度统计 ---
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = r < min(h, w) * 0.15
    high_mask = r > min(h, w) * 0.4
    feats += [float(mag[low_mask].mean()), float(mag[high_mask].mean()),
              float(mag[low_mask].std()), float(mag[high_mask].std())]
    # --- 每通道颜色统计 ---
    for c in range(3):
        ch = img[..., c]
        feats += [float(ch.mean()), float(ch.std()), float(ch.max() - ch.min())]
    return np.array(feats, dtype=np.float32)


def extract_pixel_features(image: np.ndarray) -> np.ndarray:
    """空域/像素统计特征向量（15 维）。

    组成：每通道均值/标准差/极差/95%-5% 分位差（12 维）+ 灰度图水平/垂直
    梯度绝对均值与灰度标准差（3 维）。捕捉颜色偏移与纹理/边缘统计差异。
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gray = np.mean(img, axis=2)
    feats: list = []
    for c in range(3):
        ch = img[..., c]
        feats += [float(ch.mean()), float(ch.std()),
                  float(ch.max() - ch.min()),
                  float(np.percentile(ch, 95) - np.percentile(ch, 5))]
    gx = np.diff(gray, axis=0)
    gy = np.diff(gray, axis=1)
    feats += [float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy))),
              float(np.std(gray))]
    return np.array(feats, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 检测器基类：特征 + 标准化 + 逻辑回归
# --------------------------------------------------------------------------- #
class FeatureDetector:
    """基于"特征 + StandardScaler + LogisticRegression"的轻量检测器基类。

    与 DeepfakeBench 的检测协议一致：图像先归一化（本实现以特征替代端到端
    网络），输出为 fake_probability ∈ [0,1]（1 = DeepFake）。
    """

    feature_dim = 0

    def __init__(self, C: float = 1.0, max_iter: int = 500, seed: int = 42) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        self._fitted = False

    # -- 子类需实现 --
    def _extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- 训练 --
    def fit(self, images: np.ndarray, labels: np.ndarray) -> "FeatureDetector":
        """在 (images, labels) 上拟合检测器。images: [N, H, W, 3] uint8。"""
        labels = np.asarray(labels, dtype=np.int64)
        X = np.stack([self._extract(im) for im in images])
        if len(np.unique(labels)) < 2:
            self._fitted = False
            return self
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, labels)
        self._fitted = True
        return self

    # -- 推理 --
    def predict(self, image: np.ndarray) -> float:
        """返回 P(fake | image) ∈ [0, 1]。"""
        if not self._fitted:
            return 0.5
        feat = self._extract(image).reshape(1, -1)
        feat = self._scaler.transform(feat)
        df = float(self._clf.decision_function(feat)[0])
        return float(1.0 / (1.0 + np.exp(-df)))

    def predict_batch(self, images) -> list:
        return [self.predict(im) for im in images]


# [IMPORTANT] Replace this line with real attack/defense code.

class PixelStatDefense(FeatureDetector):
    """像素统计检测器（防御固定池 D0 成员，空域特征家族）。

    特征：15 维通道统计 + 梯度/灰度统计（见 extract_pixel_features）。
    """

    name = "pixel_stat"
    feature_dim = 15

    def _extract(self, image: np.ndarray) -> np.ndarray:
        return extract_pixel_features(image)


# --------------------------------------------------------------------------- #
# 交付 Baseline 接口（赛题 §7.4：defense.py 的 defend(request) -> {"fake_probability"}）
# --------------------------------------------------------------------------- #
# 全局单例：首次调用前需先 fit（test.py 负责在训练集上拟合）
_DETECTOR = Defense()


def set_detector(detector) -> None:
    """设置当前评测使用的检测器实例（test.py 在拟合后调用）。"""
    global _DETECTOR
    _DETECTOR = detector


def get_detector():
    return _DETECTOR


def defend(request: dict) -> dict:
    """DeepFake 检测（交付 Baseline）。

    参数
    ----
    request : {"sample_id": str, "image": RGB uint8 H×W×3}

    返回
    ----
    {"fake_probability": float ∈ [0,1]，0=高置信真实，1=高置信 DeepFake}
    """
    image = request["image"]
    p = float(_DETECTOR.predict(image))
    p = min(1.0, max(0.0, p))  # 钳制到 [0,1]
    return {"fake_probability": p}


def build_defense(name: str, **params):
    """按名称构造检测器（供 test.py 使用）。"""
    table = {"frequency_stat": Defense, "pixel_stat": PixelStatDefense}
    if name not in table:
        raise ValueError(f"未知检测器: {name}")
    return table[name](**params)


# 防御固定池 D0（交付 Baseline 为 frequency_stat；pixel_stat 用于衡量攻击迁移）
FIXED_DEFENSE_POOL = [
    ("frequency_stat", {"C": 1.0, "max_iter": 500, "seed": 42}),
    ("pixel_stat", {"C": 1.0, "max_iter": 500, "seed": 42}),
]


if __name__ == "__main__":
    from sklearn import metrics as skm

    parser = argparse.ArgumentParser(description="频域统计检测器自检（交付 Baseline）")
    parser.add_argument("--detector", default="frequency_stat",
                        choices=["frequency_stat", "pixel_stat"])
    args = parser.parse_args()

    train, test = load_benchmark()
    det = build_defense(args.detector)
    det.fit(train["images"], train["labels"])
    set_detector(det)  # 使 defend() 使用已拟合的检测器

    scores = np.array([det.predict(im) for im in test["images"]])
    labels = test["labels"]
    auc = skm.roc_auc_score(labels, scores)
    acc = float(np.mean((scores >= 0.5).astype(int) == labels))
    print(f"[defense] {det.name} 检测器自检：训练 {len(train['labels'])} 张, "
          f"测试 {len(labels)} 张")
    print(f"[defense] CleanAUC = {auc:.4f}  CleanACC(τ=0.5) = {acc:.4f}")
    print("[defense] 示例推理：")
    for i in (0, 1):
        out = defend({"sample_id": test["sample_ids"][i], "image": test["images"][i]})
        print(f"          {test['sample_ids'][i]}  label={labels[i]}  "
              f"fake_probability={out['fake_probability']:.4f}")
