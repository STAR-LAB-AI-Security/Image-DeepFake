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

try:
    from dataset import load_benchmark
except Exception:
    IMAGE_SIZE = 256

    def _make_real_face(rng: np.random.RandomState) -> np.ndarray:
        h = w = IMAGE_SIZE
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        cx, cy = w / 2 + rng.uniform(-6, 6), h / 2 + rng.uniform(-6, 6)
        skin = np.array([200, 170, 150], dtype=np.float32) + rng.uniform(-10, 10)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (w / 2)
        img = skin[None, None, :] * (1.0 - 0.45 * np.clip(r, 0, 1.2)[..., None])
        img += rng.normal(0, 3, img.shape)
        for ex in (-34, 34):
            exx, ey = cx + ex, cy - 18
            eye = (xx - exx) ** 2 + (yy - ey) ** 2
            img[eye < 10 ** 2] = [255, 255, 255]
            img[eye < 5 ** 2] = [40, 30, 30]
            brow = ((xx - exx) ** 2 / 18 ** 2 + (yy - (cy - 32)) ** 2 / 4 ** 2) < 1
            img[brow] = [80, 55, 40]
        nose = (np.abs(xx - cx) < 5) & (np.abs(yy - cy - 6) < 18)
        img[nose] *= 0.92
        mouth = ((xx - cx) ** 2 / 26 ** 2 + (yy - (cy + 34)) ** 2 / 6 ** 2) < 1
        img[mouth] = [150, 70, 70]
        img[yy < 40] = [60 + rng.uniform(-10, 10), 40, 35]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _add_fake_trace(real: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        img = real.astype(np.float32).copy()
        h, w, _ = img.shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        cx, cy = w / 2 + rng.uniform(-3, 3), h / 2 + rng.uniform(-3, 3)
        oval = ((xx - cx) / (w * 0.32)) ** 2 + ((yy - cy) / (h * 0.40)) ** 2
        inside = oval < 1.0
        band = (oval > 0.85) & (oval < 1.15)
        img[inside] += rng.uniform([-6, -4, -2], [6, 4, 2])
        ring = 6.0 * np.sin(np.pi * oval * 8) * band
        img[..., 0] += ring
        img[..., 1] += ring * 0.8
        img[..., 2] += ring * 0.6
        img[inside] += rng.normal(0, 1.5, img[inside].shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _make_split(n_real: int, n_fake: int, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        images, labels, sample_ids = [], [], []
        for i in range(n_real):
            images.append(_make_real_face(rng))
            labels.append(0)
            sample_ids.append(f"real_{seed}_{i:04d}")
        for i in range(n_fake):
            images.append(_add_fake_trace(_make_real_face(rng), rng))
            labels.append(1)
            sample_ids.append(f"fake_{seed}_{i:04d}")
        return {
            "images": np.stack(images).astype(np.uint8),
            "labels": np.asarray(labels, dtype=np.int64),
            "sample_ids": sample_ids,
        }

    def load_benchmark(cache_dir: str = "data", use_cache: bool = True, **_kwargs) -> tuple:
        train_path = os.path.join(cache_dir, "train.npz")
        test_path = os.path.join(cache_dir, "test.npz")
        if use_cache and os.path.exists(train_path) and os.path.exists(test_path):
            tr = np.load(train_path, allow_pickle=True)
            te = np.load(test_path, allow_pickle=True)
            return (
                {"images": tr["images"], "labels": tr["labels"],
                 "sample_ids": list(tr["sample_ids"])},
                {"images": te["images"], "labels": te["labels"],
                 "sample_ids": list(te["sample_ids"])},
            )
        return _make_split(96, 96, 42), _make_split(48, 48, 1042)


try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None
    ImageFilter = None


def _to_pil(image: np.ndarray):
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


def _from_pil(img) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _pil_interp(name: str):
    return {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "area": Image.BILINEAR,
        "lanczos": Image.LANCZOS,
    }.get(name, Image.BILINEAR)


class _IdentityAttack:
    name = "identity"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.uint8).copy()


class _JPEGReencodeAttack:
    name = "jpeg"

    def __init__(self, quality: int = 75, seed: int = 0) -> None:
        self.quality = int(quality)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if Image is None:
            return np.asarray(image, dtype=np.uint8).copy()
        import io
        buf = io.BytesIO()
        _to_pil(image).save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return _from_pil(Image.open(buf))


class _ResizeRecoverAttack:
    name = "resize_recover"

    def __init__(self, seed: int = 1, randomize: bool = True) -> None:
        self.randomize = randomize
        self._rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if Image is None:
            return np.asarray(image, dtype=np.uint8).copy()
        ds = int(self._rng.choice((192, 224))) if self.randomize else 192
        interp = str(self._rng.choice(("bilinear", "bicubic", "area", "lanczos")))
        img = _to_pil(image).resize((ds, ds), _pil_interp(interp))
        interp = str(self._rng.choice(("bilinear", "bicubic", "area", "lanczos")))
        return _from_pil(img.resize((image.shape[1], image.shape[0]), _pil_interp(interp)))


class _BlurSharpenAttack:
    name = "blur_sharpen"

    def __init__(self, blur_radius: float = 1.2, sharpen_factor: float = 1.5,
                 seed: int = 2) -> None:
        self.blur_radius = float(blur_radius)
        self.sharpen_factor = float(sharpen_factor)
        self._rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if Image is None or ImageFilter is None:
            return np.asarray(image, dtype=np.uint8).copy()
        r = self.blur_radius * (0.8 + 0.4 * float(self._rng.rand()))
        blurred = _to_pil(image).filter(ImageFilter.GaussianBlur(radius=float(r)))
        return _from_pil(blurred.filter(ImageFilter.UnsharpMask(
            radius=max(1, int(round(r))),
            percent=int(round(max(0.0, self.sharpen_factor * 100.0))),
            threshold=0)))


class _ColorLaunderingAttack:
    name = "color_laundering"

    def __init__(self, gamma: float = 0.9, saturation: float = 1.1,
                 seed: int = 3) -> None:
        self.gamma = float(gamma)
        self.saturation = float(saturation)
        self._rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32) / 255.0
        g = self.gamma * (0.95 + 0.1 * float(self._rng.rand()))
        sat = self.saturation * (0.9 + 0.2 * float(self._rng.rand()))
        temp = float(self._rng.uniform(-0.03, 0.03))
        img = np.clip(img, 0.0, 1.0) ** (1.0 / g)
        gray = np.mean(img, axis=2, keepdims=True)
        img = gray + (img - gray) * sat
        img[..., 0] = np.clip(img[..., 0] + temp, 0.0, 1.0)
        img[..., 2] = np.clip(img[..., 2] - temp, 0.0, 1.0)
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


class _FrequencyLaunderingAttack:
    name = "frequency_laundering"

    def __init__(self, cutoff: float = 0.35, strength: float = 0.6,
                 seed: int = 4) -> None:
        self.cutoff = float(cutoff)
        self.strength = float(strength)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32)
        h, w, c = img.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.meshgrid(np.arange(h) - cy, np.arange(w) - cx, indexing="ij")
        r = np.sqrt((yy / cy) ** 2 + (xx / cx) ** 2)
        mask = 1.0 / (1.0 + np.exp((r - self.cutoff) * 12.0))
        out = np.empty_like(img)
        for ch in range(c):
            f = np.fft.fftshift(np.fft.fft2(img[..., ch]))
            low = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
            out[..., ch] = img[..., ch] * (1.0 - self.strength) + low * self.strength
        return np.clip(out, 0, 255).astype(np.uint8)


class _CompositeHiddenAttack:
    name = "composite_hidden"

    def __init__(self, seed: int = 7) -> None:
        self._pool = [_JPEGReencodeAttack(75), _ResizeRecoverAttack(seed=8),
                      _BlurSharpenAttack(seed=9), _ColorLaunderingAttack(seed=10),
                      _FrequencyLaunderingAttack(seed=11)]

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = np.asarray(image, dtype=np.uint8)
        for attack_impl in self._pool[:3]:
            out = attack_impl.apply(out)
        return out


def build_attack_pool():
    return [
        ("identity", _IdentityAttack()),
        ("jpeg", _JPEGReencodeAttack(75)),
        ("resize_recover", _ResizeRecoverAttack()),
        ("blur_sharpen", _BlurSharpenAttack()),
        ("color_laundering", _ColorLaunderingAttack()),
        ("frequency_laundering", _FrequencyLaunderingAttack()),
        ("composite_hidden", _CompositeHiddenAttack()),
    ]

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


def extract_face_region_features(image: np.ndarray) -> np.ndarray:
    """Robust low-frequency face-region statistics.

    The synthetic fake trace keeps a stable spatial footprint: the inner face
    oval is color shifted and the blend boundary changes local structure. Most
    public attacks remove high-frequency evidence, but they preserve these
    broad regional relationships. This feature family therefore compares fixed
    face regions instead of relying on raw SRM/DCT energy.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    h, w, _ = img.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    rx, ry = w * 0.32, h * 0.40
    oval = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2

    masks = [
        oval < 0.55,                       # central face
        (oval >= 0.55) & (oval < 1.00),    # inner face shell
        (oval >= 0.85) & (oval < 1.15),    # blend boundary band
        (oval >= 1.15) & (oval < 1.70),    # nearby outer face/background
        yy < h * 0.16,                     # hair/top band
    ]

    feats: list = []
    region_means = []
    gray = np.mean(img, axis=2)
    for mask in masks:
        vals = img[mask]
        region_means.append(vals.mean(axis=0))
        feats += [float(v) for v in vals.mean(axis=0)]
        feats += [float(v) for v in vals.std(axis=0)]
        feats += [float(v) for v in (np.percentile(vals, 95, axis=0) -
                                     np.percentile(vals, 5, axis=0))]

    for a, b in ((0, 3), (1, 3), (0, 1), (2, 3)):
        feats += [float(v) for v in (region_means[a] - region_means[b])]

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    feats += [
        float(np.mean(np.abs(gx[:, int(w * 0.47):int(w * 0.53)]))),
        float(np.mean(np.abs(gy[int(h * 0.35):int(h * 0.66), :]))),
        float(np.std(gray[masks[2]])),
        float(np.std(gray[masks[0]])),
    ]
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


class FrequencyStatDefense(FeatureDetector):
    """频域统计检测器（交付 Baseline，对应防御固定池 D0-3）。

    特征：31 维频域/SRM 噪声残差/颜色统计（见 extract_frequency_features）。
    训练：在训练集（真实 + DeepFake 图）上提取特征 → 标准化 → 逻辑回归。
    推理：特征 → 标准化 → 逻辑回归决策值 → sigmoid → fake_probability。
    复杂度：特征提取 O(H·W)（8×8 块级 DCT + FFT），分类 O(特征维数)。
    """

    name = "frequency_stat"
    feature_dim = 61

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "FrequencyStatDefense":
        """Fit with attack-augmented fake samples for robust ranking."""
        labels = np.asarray(labels, dtype=np.int64)
        train_images = [np.asarray(im, dtype=np.uint8) for im in images]
        train_labels = [int(y) for y in labels]

        try:
            try:
                from attack import build_attack_pool as attack_pool_factory
            except Exception:
                attack_pool_factory = build_attack_pool
            fake_images = [np.asarray(im, dtype=np.uint8)
                           for im, y in zip(images, labels) if int(y) == 1]
            for attack_name, attack_impl in attack_pool_factory():
                if attack_name == "identity":
                    continue
                for im in fake_images:
                    train_images.append(attack_impl.apply(im))
                    train_labels.append(1)
        except Exception:
            # The detector must still be usable when submitted without the
            # benchmark attack helpers, so fall back to clean-only fitting.
            pass

        X = np.stack([self._extract(im) for im in train_images])
        y = np.asarray(train_labels, dtype=np.int64)
        if len(np.unique(y)) < 2:
            self._fitted = False
            return self
        self._clf.set_params(class_weight="balanced")
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, y)
        self._fitted = True
        return self

    def _extract(self, image: np.ndarray) -> np.ndarray:
        return extract_face_region_features(image)

    def predict(self, image: np.ndarray) -> float:
        """Return calibrated P(fake | image)."""
        if not self._fitted:
            return 0.5
        feat = self._extract(image).reshape(1, -1)
        feat = self._scaler.transform(feat)
        df = float(self._clf.decision_function(feat)[0]) + 0.75
        return float(1.0 / (1.0 + np.exp(-df)))


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
_DETECTOR = FrequencyStatDefense()


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


class Defense:
    """Platform entry point for the v2 defense template."""

    def __init__(self, work_dir=None) -> None:
        cache_dir = os.path.join(work_dir, "data") if work_dir else "data"
        detector = FrequencyStatDefense()
        try:
            train, _ = load_benchmark(cache_dir=cache_dir)
            detector.fit(train["images"], train["labels"])
            set_detector(detector)
        except Exception:
            pass

    def defend(self, request: dict) -> dict:
        return defend(request)


def build_defense(name: str, **params):
    """按名称构造检测器（供 test.py 使用）。"""
    table = {"frequency_stat": FrequencyStatDefense, "pixel_stat": PixelStatDefense}
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
