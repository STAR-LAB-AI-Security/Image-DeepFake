#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击代码 —— 图像痕迹隐匿与鲁棒检测基准（Baseline：JPEG 重编码）与固定攻击池 A0。

赛题（DeepFake 图像痕迹隐匿与鲁棒检测攻防对抗）中，攻击方不负责生成 DeepFake，
而是对主办方提供的 DeepFake 图像做"查询无关"的轻量后处理（痕迹隐匿）：在保持
人物身份、语义与视觉质量基本不变的前提下，降低未知检测器对伪造图像的识别能力。
攻击输入为 RGB uint8 图像（H×W×3），输出为同尺寸、同 dtype 的图像。

本文件实现赛题建议的固定攻击池 A0（材料 deepguard_backend 中的官方攻击池）：
    identity             无操作（清洁基线）
    jpeg                 JPEG 重编码（质量 ∈ {55,65,75,85}，默认 85）——交付 Baseline
    resize_recover       缩放-恢复（256 → 192/224 → 256，随机插值）
    blur_sharpen         轻度高斯模糊 + 重新锐化
    color_laundering     颜色洗白（Gamma/饱和度/色温轻微组合）
    frequency_laundering 频域洗白（FFT 低通/陷波滤波）
    composite_hidden     隐藏组合攻击（2~4 步随机链，模拟主办方隐藏攻击）

角色分工（槽位与官方基线互不耦合）：
    ① 槽位（学生提交，契约见 attack-blank.py）：class JPEGReencodeAttack
        被测攻击主体（__init__(quality=85, seed=42)、apply(image) -> np.ndarray），
        由 test.py 直接构造评测；学生可完全自定义实现。
    ② 后端自带官方基线 BaselineJPEGAttack（固定攻击池 jpeg 成员）：
        供防御鲁棒性评测（test.py --stage defense）与模块级 attack() 兼容层使用，
        与槽位学生代码无关。

用法：
    python attack.py                   # 对测试集伪造图施加官方 JPEG 隐匿，输出 data/attacked_fakes.npz
    python attack.py --quality 85 --output data/attacked_fakes.npz
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
from PIL import Image, ImageFilter

from dataset import load_benchmark, load_npz, save_npz

IMAGE_SIZE = 256

# JPEG 质量池（赛题 A0-1：quality ∈ {55, 65, 75, 85}）
JPEG_QUALITIES = (55, 65, 75, 85)
# 缩放-恢复的中间尺寸（赛题 A0-2：192/224）
DOWNSIZE_POOL = (192, 224)
# 插值方式池
INTERP_POOL = ("bilinear", "bicubic", "area", "lanczos")


# --------------------------------------------------------------------------- #
# 图像工具
# --------------------------------------------------------------------------- #
def to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


def from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _pil_interp(name: str):
    return {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "area": Image.BILINEAR,  # PIL 无 area，用 bilinear 近似
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
    }.get(name, Image.BILINEAR)


def resize_pair(image: np.ndarray, size: int, interp: str = "bilinear") -> np.ndarray:
    """等比缩放到 (size, size)（RGB uint8 -> RGB uint8）。"""
    return from_pil(to_pil(image).resize((size, size), _pil_interp(interp)))


# --------------------------------------------------------------------------- #
# 固定攻击池 A0
# --------------------------------------------------------------------------- #
class IdentityAttack:
    """无操作攻击：输出 = 输入（清洁评测基线，不属于真实隐匿手段）。"""

    name = "identity"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.uint8).copy()


class BaselineJPEGAttack:
    """JPEG 重编码攻击（官方基线 A0-1，后端自带；固定攻击池 jpeg 成员）。

    原理：JPEG 以 8×8 分块 DCT + 量化丢弃高频系数实现有损压缩。对伪造图像重新
    执行"编码 → 解码"会打乱原图的块边界、量化掉高频取证痕迹（融合边界的振铃、
    局部纹理异常），从而削弱依赖频域/高频统计的检测器。JPEG 质量越低压缩越强，
    隐匿效果越明显，但图像失真也越大，需在保真约束（SSIM/PSNR）内权衡。

    参数：
        quality    : JPEG 质量（默认 85，赛题最小示例；可从 {55,65,75,85} 池中选）
        seed       : 随机种子（默认 42，本实现为确定性重编码，种子保留接口一致性）

    复杂度：O(H·W)（逐像素编码/解码），完全查询无关、无需训练。
    """

    name = "jpeg"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        if quality not in JPEG_QUALITIES:
            # 就近收敛到允许的质量池（与后端行为一致）
            quality = int(min(JPEG_QUALITIES, key=lambda q: abs(q - quality)))
        self.quality = quality
        self.seed = seed

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = to_pil(image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return from_pil(Image.open(buf))


"""攻击方（痕迹隐匿）——JPEG 强压缩重编码（quality=55）。

契约：
1) 工程上传 / test.py：JPEGReencodeAttack(quality=85, seed=42).apply(image) -> ndarray
2) 在线编辑 / PDF：attack(sample) -> {"image": ...}

算法：真正对应类名 JPEGReencodeAttack 的 JPEG 重编码。quality=55 的强压缩
足以量化掉棋盘格高频痕迹。注意：若对方是「训练过的检测器」，JPEG 类攻击
迁移性弱于频域陷波，建议优先用 01/02。
"""
import io

import numpy as np
from PIL import Image


def _to_uint8_rgb(image):
    if isinstance(image, dict):
        image = image.get("image", image)
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return img


class JPEGReencodeAttack:
    name = "jpeg"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.quality = int(quality)
        self.seed = int(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = _to_uint8_rgb(image)
        q = 55  # 固定强压缩，充分移除棋盘格（不随传入 quality 变化）
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(
            buf, format="JPEG", quality=q, subsampling=2, optimize=False)
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
        if out.shape != img.shape:
            out = np.asarray(Image.fromarray(out).resize(
                (img.shape[1], img.shape[0]), Image.BICUBIC))
        return out


def attack(sample, quality=85, seed=42):
    image = sample["image"] if isinstance(sample, dict) else sample
    return {"image": JPEGReencodeAttack(quality=quality, seed=seed).apply(image)}


class ResizeRecoverAttack:
    """缩放-恢复攻击（A0-2）：256 → 192/224 → 256，随机插值。"""

    name = "resize_recover"

    def __init__(self, down_size: int = 192, interp: str = "bilinear",
                 randomize: bool = True, seed: int = 0) -> None:
        self.down_size = down_size
        self.interp = interp
        self.randomize = randomize
        self._rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        ds, interp = self.down_size, self.interp
        if self.randomize:
            ds = int(self._rng.choice(DOWNSIZE_POOL))
            interp = str(self._rng.choice(INTERP_POOL))
        small = resize_pair(image, ds, interp)
        recover = interp if not self.randomize else str(self._rng.choice(INTERP_POOL))
        return resize_pair(small, image.shape[1], recover)


class BlurSharpenAttack:
    """模糊-锐化攻击（A0-3）：轻度高斯模糊后重新锐化。"""

    name = "blur_sharpen"

    def __init__(self, blur_radius: float = 1.2, sharpen_factor: float = 1.5,
                 seed: int = 0) -> None:
        self.blur_radius = float(blur_radius)
        self.sharpen_factor = float(sharpen_factor)
        self._rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        r = self.blur_radius * (0.8 + 0.4 * float(self._rng.rand()))
        blurred = to_pil(image).filter(ImageFilter.GaussianBlur(radius=float(r)))
        radius_i = max(1, int(round(r)))
        amount = max(0.0, self.sharpen_factor * 100.0)
        sharpened = blurred.filter(ImageFilter.UnsharpMask(
            radius=radius_i, percent=int(round(amount)), threshold=0))
        return from_pil(sharpened)


class ColorLaunderingAttack:
    """颜色洗白攻击（A0-4）：轻微 Gamma + 饱和度 + 色温组合。"""

    name = "color_laundering"

    def __init__(self, gamma: float = 0.9, saturation: float = 1.1,
                 temperature: float = 0.0, seed: int = 0) -> None:
        self.gamma = float(gamma)
        self.saturation = float(saturation)
        self.temperature = float(temperature)
        self._rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32) / 255.0
        g = self.gamma * (0.95 + 0.1 * float(self._rng.rand()))
        sat = self.saturation * (0.9 + 0.2 * float(self._rng.rand()))
        temp = self.temperature + float(self._rng.uniform(-0.03, 0.03))
        img = np.clip(img, 0.0, 1.0) ** (1.0 / g)           # gamma
        gray = np.mean(img, axis=2, keepdims=True)
        img = gray + (img - gray) * sat                      # 饱和度
        img[..., 0] = np.clip(img[..., 0] + temp, 0.0, 1.0)  # 色温（暖 +R -B）
        img[..., 2] = np.clip(img[..., 2] - temp, 0.0, 1.0)
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


class FrequencyLaunderingAttack:
    """频域洗白攻击（A0-5）：FFT 低通/陷波滤波，抑制高频取证痕迹。"""

    name = "frequency_laundering"

    def __init__(self, cutoff: float = 0.35, strength: float = 0.6,
                 mode: str = "lowpass", seed: int = 0) -> None:
        self.cutoff = float(cutoff)
        self.strength = float(strength)
        self.mode = mode
        self._rng = np.random.RandomState(seed)

    def _build_mask(self, h: int, w: int) -> np.ndarray:
        cy, cx = h // 2, w // 2
        y = np.arange(h) - cy
        x = np.arange(w) - cx
        yy, xx = np.meshgrid(y, x, indexing="ij")
        r = np.sqrt((yy / cy) ** 2 + (xx / cx) ** 2)  # 归一化半径
        if self.mode == "lowpass":
            mask = 1.0 / (1.0 + np.exp((r - self.cutoff) * 12.0))  # 平滑低通
        elif self.mode == "notch":
            mask = np.ones_like(r)
            band = (r > self.cutoff) & (r < self.cutoff + 0.2)
            mask[band] = 1.0 - self.strength
        else:
            mask = np.ones_like(r)
        return mask

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32)
        h, w, c = img.shape
        mask = self._build_mask(h, w)
        out = np.empty_like(img)
        for ch in range(c):
            f = np.fft.fftshift(np.fft.fft2(img[..., ch]))
            f = f * mask
            low = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
            out[..., ch] = img[..., ch] * (1.0 - self.strength) + low * self.strength
        return np.clip(out, 0, 255).astype(np.uint8)


class CompositeHiddenAttack:
    """隐藏组合攻击（A0-Hidden）：从公开攻击中随机抽取 2~4 步串联。

    模拟主办方构造的未公开组合攻击。链的种子固定（默认 7），可复现但不公开。
    """

    name = "composite_hidden"

    def __init__(self, steps: int = 3, seed: int = 7, quality: int = 75) -> None:
        self.steps = int(steps)
        self.quality = quality
        self._rng = np.random.RandomState(seed)
        self._pool = [
            ("jpeg", lambda s: BaselineJPEGAttack(
                quality=int(s.choice(list(JPEG_QUALITIES))), seed=int(s.randint(10 ** 6)))),
            ("resize_recover", lambda s: ResizeRecoverAttack(randomize=True, seed=int(s.randint(10 ** 6)))),
            ("blur_sharpen", lambda s: BlurSharpenAttack(seed=int(s.randint(10 ** 6)))),
            ("color_laundering", lambda s: ColorLaunderingAttack(seed=int(s.randint(10 ** 6)))),
            ("frequency_laundering", lambda s: FrequencyLaunderingAttack(seed=int(s.randint(10 ** 6)))),
        ]

    def apply(self, image: np.ndarray) -> np.ndarray:
        n = max(2, min(4, self.steps))
        idxs = list(range(len(self._pool)))
        self._rng.shuffle(idxs)
        chosen = [self._pool[i] for i in idxs[:n]]
        out = np.asarray(image, dtype=np.uint8)
        for _name, factory in chosen:
            out = factory(self._rng).apply(out)
        return out


# --------------------------------------------------------------------------- #
# 交付 Baseline 接口（赛题 §5.4：attack.py 的 attack(sample) -> {"image"}）
# --------------------------------------------------------------------------- #
def attack(sample: dict, quality: int = 85, seed: int = 42) -> dict:
    """JPEG 重编码攻击（交付 Baseline）。

    参数
    ----
    sample  : {"sample_id": str, "image": RGB uint8 H×W×3}
    quality : JPEG 质量（默认 85）
    seed    : 随机种子（默认 42）

    返回
    ----
    {"image": RGB uint8 H×W×3，与输入同尺寸}
    """
    image = sample["image"]
    out = BaselineJPEGAttack(quality=quality, seed=seed).apply(image)
    return {"image": out}


def build_attack(name: str, **params):
    """按名称构造官方固定攻击池成员（供 test.py 使用，与槽位学生代码无关）。"""
    table = {
        "identity": IdentityAttack,
        "jpeg": BaselineJPEGAttack,
        "resize_recover": ResizeRecoverAttack,
        "blur_sharpen": BlurSharpenAttack,
        "color_laundering": ColorLaunderingAttack,
        "frequency_laundering": FrequencyLaunderingAttack,
        "composite_hidden": CompositeHiddenAttack,
    }
    if name not in table:
        raise ValueError(f"未知攻击: {name}")
    return table[name](**params)


# 固定攻击池 A0（与赛题材料 deepguard_backend/config.yaml 的攻击池一致）
FIXED_ATTACK_POOL = [
    ("identity", {}),
    ("jpeg", {"quality": 75, "random_quality": True, "seed": 0}),
    ("resize_recover", {"randomize": True, "seed": 1}),
    ("blur_sharpen", {"blur_radius": 1.2, "sharpen_factor": 1.5, "seed": 2}),
    ("color_laundering", {"gamma": 0.9, "saturation": 1.1, "seed": 3}),
    ("frequency_laundering", {"cutoff": 0.35, "strength": 0.6, "mode": "lowpass", "seed": 4}),
    ("composite_hidden", {"steps": 3, "seed": 7}),
]


def _jpeg_attack_with_random(quality: int = 75, random_quality: bool = False, seed: int = 0):
    """兼容固定池参数：random_quality=True 时随机选择质量。"""
    rng = np.random.RandomState(seed)
    q = quality
    if random_quality:
        q = int(rng.choice(JPEG_QUALITIES))
    return BaselineJPEGAttack(quality=q, seed=seed)


def build_attack_pool(pool=None):
    """构造固定攻击池 A0 的实例列表（含参数），返回 [(name, instance), ...]。"""
    out = []
    for name, params in (pool or FIXED_ATTACK_POOL):
        if name == "jpeg":
            # jpeg 支持 random_quality 参数（False 时固定质量）
            out.append((name, _jpeg_attack_with_random(**params)))
        else:
            out.append((name, build_attack(name, **params)))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG 重编码攻击（交付 Baseline）")
    parser.add_argument("--quality", type=int, default=85, help="JPEG 质量，默认 85")
    parser.add_argument("--input", default=None,
                        help="输入 npz（默认取测试集伪造图子集）")
    parser.add_argument("--output", default="data/attacked_fakes.npz", help="输出路径")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input:
        data = load_npz(args.input)
        fake_mask = data["labels"] == 1
        images = data["images"][fake_mask]
        labels = data["labels"][fake_mask]
        ids = [data["sample_ids"][i] for i in range(len(data["labels"])) if data["labels"][i] == 1] \
            if "sample_ids" in data else None
    else:
        _tr, test = load_benchmark()
        fake_mask = test["labels"] == 1
        images = test["images"][fake_mask]
        labels = test["labels"][fake_mask]
        ids = [test["sample_ids"][i] for i in range(len(test["labels"])) if test["labels"][i] == 1]

    atk = BaselineJPEGAttack(quality=args.quality, seed=args.seed)
    attacked = np.stack([atk.apply(im) for im in images])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_npz(args.output, attacked, labels, ids)
    # 简单的合法性自检：尺寸/通道/取值
    same_shape = attacked.shape == images.shape
    print(f"[attack] JPEG 重编码完成：{len(labels)} 张伪造图 -> {args.output}  "
          f"形状一致={same_shape} 值域 [{attacked.min()}, {attacked.max()}]")
