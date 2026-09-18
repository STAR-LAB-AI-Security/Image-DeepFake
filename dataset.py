#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集代码 —— 图像痕迹隐匿与鲁棒检测基准（Benchmark）数据集处理。

职责：
1. 生成/缓存基准"合成人脸图像集"（真实图 + DeepFake 模拟图，256×256 RGB）；
2. 确定性生成（固定随机种子），支持 npz 本地缓存，无网络可跑；
3. 统一输出格式（基准输入输出规范）：

       train = {"images": [N_train, 256, 256, 3] uint8 RGB（取值 0~255）,
                "labels": [N_train] int64，{0=real, 1=fake},
                "sample_ids": [N_train] str}
       test  = {"images": [N_test, 256, 256, 3] uint8,
                "labels": [N_test] int64,
                "sample_ids": [N_test] str}

    基准固定划分为训练 192 张（96 真 + 96 假）与测试 96 张（48 真 + 48 假），
    随机种子 42，划分完全确定；测试集在整个评测过程中保持干净、不参与训练，
    模拟"隐藏测试集"。

数据形态忠实于赛题（赛题以 FaceForensics++ 等真实人脸 DeepFake 数据集为主数据
源，统一人脸检测/对齐/裁剪到 256×256 后送入攻击与防御程序）：
- 真实图：程序化生成的"类人脸"图像（肤色渐变底 + 眼睛/眉毛/鼻子/嘴/发带 +
  高斯纹理噪声），对应裁剪后的真实人脸；
- 伪造图：在真实图基础上注入"合成融合边界痕迹"（人脸内部颜色偏移 + 边界高频
  振铃 + 内部纹理噪声），模拟 DeepFake 生成过程留下的取证痕迹——这类痕迹可被
  频域特征（DCT/FFT/SRM 噪声残差）与空域/颜色特征捕获，也可被后处理隐匿攻击
  削弱，与赛题中 DeepFake 痕迹的行为一致。

用法：
    python dataset.py                 # 生成/加载基准数据并打印规模
"""


import os

import numpy as np

# --------------------------------------------------------------------------- #
# 常量（与基准文档一致）
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 256
DEFAULT_TRAIN_REAL = 96
DEFAULT_TRAIN_FAKE = 96
DEFAULT_TEST_REAL = 48
DEFAULT_TEST_FAKE = 48
DEFAULT_SEED = 42
CACHE_DIR = "data"


# --------------------------------------------------------------------------- #
# 合成人脸图像生成（忠实于赛题材料 deepguard_backend 的 synthetic 数据生成器）
# --------------------------------------------------------------------------- #
def _make_real_face(rng: np.random.RandomState) -> np.ndarray:
    """程序化生成一张 256×256 的"类人脸"真实图像（RGB uint8）。

    结构：肤色径向渐变底 + 眼睛/眉毛/鼻子/嘴 + 顶部发带 + 高斯纹理噪声。
    """
    h = w = IMAGE_SIZE
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w / 2 + rng.uniform(-6, 6), h / 2 + rng.uniform(-6, 6)

    # 肤色底：径向渐变（中心更亮）
    base_hue = rng.uniform(-10, 10)
    skin = np.array([200, 170, 150], dtype=np.float32) + base_hue
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (w / 2)
    r = np.clip(r, 0, 1.2)
    img = skin[None, None, :] * (1.0 - 0.45 * r[..., None])
    img += rng.normal(0, 3, img.shape)

    # 眼睛
    for ex in (-34, 34):
        exx = cx + ex
        ey = cy - 18
        eye = (xx - exx) ** 2 + (yy - ey) ** 2
        img[eye < 10 ** 2] = [255, 255, 255]
        img[eye < 5 ** 2] = [40, 30, 30]
    # 眉毛
    for ex in (-34, 34):
        exx = cx + ex
        ey = cy - 32
        m = ((xx - exx) ** 2 / 18 ** 2 + (yy - ey) ** 2 / 4 ** 2) < 1
        img[m] = [80, 55, 40]
    # 鼻子
    m = (np.abs(xx - cx) < 5) & (np.abs(yy - cy - 6) < 18)
    img[m] *= 0.92
    # 嘴
    my = cy + 34
    m = ((xx - cx) ** 2 / 26 ** 2 + (yy - my) ** 2 / 6 ** 2) < 1
    img[m] = [150, 70, 70]
    # 顶部发带
    m = yy < 40
    img[m] = [60 + rng.uniform(-10, 10), 40, 35]

    return np.clip(img, 0, 255).astype(np.uint8)


def _add_fake_trace(real: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """在真实图上注入"合成融合边界"伪造痕迹，生成 DeepFake 模拟图。

    痕迹组成（对应真实 DeepFake 的换脸融合痕迹）：
    1. 人脸内部（椭圆区域）颜色偏移（每通道独立随机偏移）；
    2. 边界带内的高频振铃（模拟融合边界/重采样振铃伪影）；
    3. 内部轻度高频噪声（模拟生成网络的上采样伪影）。
    """
    img = real.astype(np.float32).copy()
    h, w, _ = img.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w / 2 + rng.uniform(-3, 3), h / 2 + rng.uniform(-3, 3)
    rx, ry = w * 0.32, h * 0.40
    oval = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    inside = oval < 1.0
    # 颜色偏移（模拟换脸区域色差）
    band = (oval > 0.85) & (oval < 1.15)
    shift = rng.uniform([-6, -4, -2], [6, 4, 2])
    img[inside] += shift
    # 边界高频振铃
    ring = 6.0 * np.sin(np.pi * oval * 8) * band
    img[..., 0] += ring
    img[..., 1] += ring * 0.8
    img[..., 2] += ring * 0.6
    # 内部高频噪声
    img[inside] += rng.normal(0, 1.5, img[inside].shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_split(n_real: int, n_fake: int, seed: int) -> dict:
    """生成一个划分：n_real 张真实图 + n_fake 张伪造图（确定性）。"""
    rng = np.random.RandomState(seed)
    images, labels, sample_ids = [], [], []
    for i in range(n_real):
        images.append(_make_real_face(rng))
        labels.append(0)
        sample_ids.append(f"real_{seed}_{i:04d}")
    for i in range(n_fake):
        base = _make_real_face(rng)
        images.append(_add_fake_trace(base, rng))
        labels.append(1)
        sample_ids.append(f"fake_{seed}_{i:04d}")
    images = np.stack(images).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    return {"images": images, "labels": labels, "sample_ids": sample_ids}


# --------------------------------------------------------------------------- #
# 基准数据加载
# --------------------------------------------------------------------------- #
def load_benchmark(
    train_real: int = DEFAULT_TRAIN_REAL,
    train_fake: int = DEFAULT_TRAIN_FAKE,
    test_real: int = DEFAULT_TEST_REAL,
    test_fake: int = DEFAULT_TEST_FAKE,
    seed: int = DEFAULT_SEED,
    cache_dir: str = CACHE_DIR,
    use_cache: bool = True,
) -> tuple[dict, dict]:
    """加载基准训练集与测试集，返回 (train, test)，各为 {images, labels, sample_ids}。

    首次调用生成数据并落盘 data/train.npz、data/test.npz；之后直接读取本地
    npz 缓存，保证划分完全一致、结果可复现。
    """
    train_path = os.path.join(cache_dir, "train.npz")
    test_path = os.path.join(cache_dir, "test.npz")

    if use_cache and os.path.exists(train_path) and os.path.exists(test_path):
        t = np.load(train_path, allow_pickle=True)
        e = np.load(test_path, allow_pickle=True)
        train = {"images": t["images"], "labels": t["labels"],
                 "sample_ids": list(t["sample_ids"])}
        test = {"images": e["images"], "labels": e["labels"],
                "sample_ids": list(e["sample_ids"])}
        print(f"[dataset] 命中本地缓存: {train_path}, {test_path}")
        return train, test

    train = _make_split(train_real, train_fake, seed)
    test = _make_split(test_real, test_fake, seed + 1000)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        train_path,
        images=train["images"], labels=train["labels"],
        sample_ids=np.asarray(train["sample_ids"], dtype=object),
    )
    np.savez_compressed(
        test_path,
        images=test["images"], labels=test["labels"],
        sample_ids=np.asarray(test["sample_ids"], dtype=object),
    )
    print(f"[dataset] 已生成并保存基准数据缓存: {train_path}, {test_path}")
    return train, test


def save_npz(path: str, images: np.ndarray, labels: np.ndarray,
             sample_ids=None) -> None:
    """将 {images, labels, sample_ids} 数据保存为 .npz（供攻击/防御产物落盘）。"""
    d = {"images": np.asarray(images, dtype=np.uint8),
         "labels": np.asarray(labels, dtype=np.int64)}
    if sample_ids is not None:
        d["sample_ids"] = np.asarray(list(sample_ids), dtype=object)
    np.savez_compressed(path, **d)
    print(f"[dataset] 已保存 {path}（{len(d['labels'])} 个样本）")


def load_npz(path: str) -> dict:
    """读取 .npz 文件为 {images, labels, sample_ids} 字典。"""
    d = np.load(path, allow_pickle=True)
    out = {"images": d["images"], "labels": d["labels"]}
    if "sample_ids" in d:
        out["sample_ids"] = list(d["sample_ids"])
    return out


if __name__ == "__main__":
    tr, te = load_benchmark()
    print(f"train.images: {tr['images'].shape}  dtype={tr['images'].dtype}  "
          f"值域 [{tr['images'].min()}, {tr['images'].max()}]")
    print(f"train.labels: {tr['labels'].shape}  真={int((tr['labels']==0).sum())} "
          f"假={int((tr['labels']==1).sum())}")
    print(f"test.images:  {te['images'].shape}  test.labels: {te['labels'].shape}  "
          f"真={int((te['labels']==0).sum())} 假={int((te['labels']==1).sum())}")
    print(f"示例样本 id: {te['sample_ids'][0]}, {te['sample_ids'][-1]}")
