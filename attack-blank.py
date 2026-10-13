class JPEGReencodeAttack:
    """Official v2 interface; implementation is radial correction + shared field.

    Frozen parameters: grid5, rho1, detail0.60, amplitude4; SSIM>=.93, PSNR>=30.
    Reads only current pixels and explicit seed. No detector, labels, ID, or files.
    The required class name is retained for backend compatibility.
    """
    name = "jpeg"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.quality = quality
        self.seed = seed
        self._engine = self._make_engine(seed)

    def apply(self, image):
        return self._engine.apply(image)

    @staticmethod
    def _make_engine(seed):
        import io
        import math
        import hashlib
        from functools import lru_cache
        import numpy as np
        from PIL import Image
        from scipy.ndimage import gaussian_filter, uniform_filter
        from scipy.optimize import least_squares
        from threadpoolctl import threadpool_limits

        def _validate_rgb(image: np.ndarray) -> np.ndarray:
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
                raise ValueError('image 必须是非空 H×W×3 RGB 图像')
            if image.dtype != np.uint8:
                raise TypeError('image 的 dtype 必须为 uint8，像素范围为 0~255')
            return image

        def _window_mean(x: np.ndarray) -> np.ndarray:
            return gaussian_filter(x, sigma=(1.5, 1.5, 0), radius=(5, 5, 0), mode='reflect')

        class _FidelityReference:
            """缓存原图局部统计，供同一张图的多次保真检查复用。"""

            def __init__(self, image: np.ndarray) -> None:
                self.image = image.astype(np.float64)
                self.mean = _window_mean(self.image)
                self.variance = _window_mean(self.image ** 2) - self.mean ** 2

            def measure(self, candidate: np.ndarray) -> tuple[float, float]:
                other = candidate.astype(np.float64)
                other_mean = _window_mean(other)
                other_var = _window_mean(other ** 2) - other_mean ** 2
                covariance = _window_mean(self.image * other) - self.mean * other_mean
                c1, c2 = ((0.01 * 255) ** 2, (0.03 * 255) ** 2)
                numerator = (2 * self.mean * other_mean + c1) * (2 * covariance + c2)
                denominator = (self.mean ** 2 + other_mean ** 2 + c1) * (self.variance + other_var + c2)
                similarity = float(np.mean(numerator / np.maximum(denominator, 1e-12)))
                mse = float(np.mean((self.image - other) ** 2))
                psnr = 99.0 if mse <= 1e-12 else float(10 * np.log10(255 ** 2 / mse))
                return (similarity, psnr)

        def _flat_region_ratio(image: np.ndarray) -> float:
            """使用原评测器的 8×8 网格与颜色量化规则，防止新增大面积纯色。"""
            h, w, _ = image.shape
            q = (image.astype(np.int16) >> 4).astype(np.int16)
            codes = q[..., 0] * 4096 + q[..., 1] * 64 + q[..., 2]
            ys = np.linspace(0, h, 9).astype(int)
            xs = np.linspace(0, w, 9).astype(int)
            flat, total = (0, 0)
            for top, bottom in zip(ys[:-1], ys[1:]):
                for left, right in zip(xs[:-1], xs[1:]):
                    block = codes[top:bottom, left:right]
                    if block.size:
                        total += 1
                        flat += int(block.std() < 1.0)
            return flat / max(total, 1)

        class AdaptiveLaunderingAttack:
            """固定后处理链 + 仅依据图像保真度的强度回退。

            1. sigma=0.6 的轻度高斯平滑，衰减局部高频噪声。
            2. 双三次插值缩至原宽高的 7/8，再恢复，扰动重采样痕迹。
            3. JPEG quality=85，4:4:4 色度采样，减少额外的色度失真。
            4. 输出 round(I + alpha * (T(I)-I))；二分回退 alpha，保存已实测
               满足 SSIM/PSNR 的版本。所有检查都在最终 uint8 图上进行。

            不以检测器分数搜索参数；同一图像的结果不受调用顺序或 ID 影响。
            默认保真下限比题目留出余量：SSIM≥0.93，PSNR≥30dB。
            seed 为兼容接口保留，当前算法没有随机步骤。
            """
            name = 'adaptive_laundering'

            def __init__(self, quality: int=85, seed: int=42, min_ssim: float=0.93, min_psnr: float=30.0, search_steps: int=8) -> None:
                if quality not in (55, 65, 75, 85):
                    raise ValueError('quality 必须取自题目质量池 {55, 65, 75, 85}')
                if not np.isfinite(min_ssim) or not 0.92 <= min_ssim <= 1:
                    raise ValueError('min_ssim 必须在 [0.92, 1] 内')
                if not np.isfinite(min_psnr) or not 28 <= min_psnr <= 99:
                    raise ValueError('min_psnr 必须在 [28, 99] 内')
                if not isinstance(search_steps, int) or not 1 <= search_steps <= 16:
                    raise ValueError('search_steps 必须是 1~16 的整数')
                self.quality = int(quality)
                self.seed = int(seed)
                self.min_ssim = float(min_ssim)
                self.min_psnr = float(min_psnr)
                self.search_steps = search_steps

            def _transform(self, image: np.ndarray) -> np.ndarray:
                blurred = gaussian_filter(image.astype(np.float32), sigma=(0.6, 0.6, 0), mode='reflect')
                pil = Image.fromarray(np.clip(np.rint(blurred), 0, 255).astype(np.uint8))
                h, w = image.shape[:2]
                smaller = (max(1, round(w * 0.875)), max(1, round(h * 0.875)))
                resampling = getattr(Image, 'Resampling', Image)
                pil = pil.resize(smaller, resampling.BICUBIC)
                pil = pil.resize((w, h), resampling.BICUBIC)
                with io.BytesIO() as buf:
                    pil.save(buf, format='JPEG', quality=self.quality, subsampling=0)
                    buf.seek(0)
                    with Image.open(buf) as decoded:
                        return np.array(decoded.convert('RGB'), dtype=np.uint8)

            def apply_with_info(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
                """返回攻击图与保真诊断；赛题接口只返回图像，不传出诊断字段。"""
                original = _validate_rgb(image)
                candidate = self._transform(original)
                reference = _FidelityReference(original)
                similarity, psnr = reference.measure(candidate)
                alpha, checks = (1.0, 1)
                if similarity < self.min_ssim or psnr < self.min_psnr:
                    source = original.astype(np.float64)
                    delta = candidate.astype(np.float64) - source
                    best, alpha, similarity, psnr = (original.copy(), 0.0, 1.0, 99.0)
                    lo, hi = (0.0, 1.0)
                    for _ in range(self.search_steps):
                        mid = (lo + hi) / 2
                        blended = np.clip(np.rint(source + mid * delta), 0, 255).astype(np.uint8)
                        trial_ssim, trial_psnr = reference.measure(blended)
                        checks += 1
                        if trial_ssim >= self.min_ssim and trial_psnr >= self.min_psnr:
                            lo = mid
                            best, alpha = (blended, mid)
                            similarity, psnr = (trial_ssim, trial_psnr)
                        else:
                            hi = mid
                    candidate = best
                flat_fallback = _flat_region_ratio(candidate) > 0.5
                if flat_fallback:
                    candidate, alpha, similarity, psnr = (original.copy(), 0.0, 1.0, 99.0)
                return (np.ascontiguousarray(candidate), {'alpha': alpha, 'ssim': similarity, 'psnr': psnr, 'fidelity_checks': checks, 'flat_region_fallback': flat_fallback})

            def apply(self, image: np.ndarray) -> np.ndarray:
                return self.apply_with_info(image)[0]

        def _guided_base(rgb: np.ndarray, radius: int) -> np.ndarray:
            """用灰度引导的局部线性滤波保留强边缘，抑制低对比度波纹。"""
            guidance = gaussian_filter(rgb.mean(axis=2), sigma=0.8, mode='reflect')
            target = gaussian_filter(rgb, sigma=(0.8, 0.8, 0), mode='reflect')
            size = 2 * radius + 1
            mean_g = uniform_filter(guidance, size=size, mode='reflect')
            mean_p = uniform_filter(target, size=(size, size, 1), mode='reflect')
            variance = uniform_filter(guidance ** 2, size=size, mode='reflect') - mean_g ** 2
            covariance = uniform_filter(guidance[..., None] * target, size=(size, size, 1), mode='reflect') - mean_g[..., None] * mean_p
            slope = covariance / (np.maximum(variance, 0)[..., None] + 64.0)
            intercept = mean_p - slope * mean_g[..., None]
            return uniform_filter(slope, size=(size, size, 1), mode='reflect') * guidance[..., None] + uniform_filter(intercept, size=(size, size, 1), mode='reflect')

        def _smooth_field(rng: np.random.Generator, h: int, w: int, grid: int, channels: int) -> np.ndarray:
            """从低分辨率随机场生成平滑扰动，去除均值和线性照明趋势。"""
            resampling = getattr(Image, 'Resampling', Image)
            yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
            design = np.stack([np.ones(h * w), yy.ravel(), xx.ravel()], axis=1)
            field = np.empty((h, w, channels), dtype=np.float32)
            for c in range(channels):
                coarse = rng.normal(size=(grid, grid)).astype(np.float32)
                resized = np.asarray(Image.fromarray(coarse).resize((w, h), resampling.BICUBIC))
                flat = resized.astype(np.float64).ravel()
                flat -= design @ np.linalg.lstsq(design, flat, rcond=None)[0]
                field[..., c] = (flat / max(float(flat.std()), 1e-06)).reshape(h, w)
            return np.clip(field, -2.5, 2.5)

        def _radial_surface(rgb, fit_scale=1.5, shared_color=False):
            h, w = rgb.shape[:2]
            yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
            stride = max(1, round(min(h, w) / 64))
            x, y = (xx[::stride, ::stride].ravel(), yy[::stride, ::stride].ravel())
            target = gaussian_filter(rgb, (1, 1, 0), mode='reflect')[::stride, ::stride].reshape(-1, 3)
            radius = np.hypot(x, y)
            cap = np.quantile(radius, 0.95)
            design = np.stack([np.ones_like(radius), np.minimum(radius, cap)], axis=1)
            coef = np.linalg.lstsq(design, target, rcond=None)[0]
            initial = np.r_[0.0, 0.0, cap, coef[0], coef[1]]
            if shared_color:
                slope = np.dot(coef[0], coef[1]) / max(np.dot(coef[0], coef[0]), 1)
                initial = np.r_[0.0, 0.0, cap, coef[0], slope]

            def residual(p):
                r = np.minimum(np.hypot(x - p[0], y - p[1]), p[2])
                estimate = p[3:6] * (1 + r[:, None] * p[6]) if shared_color else p[3:6] + r[:, None] * p[6:9]
                return (estimate - target).ravel()
            lower = np.r_[-0.5, -0.5, 0.7, [-500.0] * (len(initial) - 3)]
            upper = np.r_[0.5, 0.5, 2.0, [500.0] * (len(initial) - 3)]
            initial = np.clip(initial, lower + 1e-06, upper - 1e-06)
            fit = least_squares(residual, initial, loss='soft_l1', f_scale=fit_scale, bounds=(lower, upper), max_nfev=100)
            p = fit.x
            radius = np.minimum(np.hypot(xx - p[0], yy - p[1]), p[2])
            return p[3:6] * (1 + radius[..., None] * p[6]) if shared_color else p[3:6] + radius[..., None] * p[6:9]

        @lru_cache(maxsize=2)
        def _components(shape, pixels):
            rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(shape).astype(np.float32)
            base = _guided_base(rgb, max(1, round(min(shape[:2]) / 32)))
            fine = rgb - gaussian_filter(rgb, (1, 1, 0), mode='reflect')
            surface = _radial_surface(rgb, fit_scale=1.5, shared_color=True)
            return (base, fine, surface)

        @lru_cache(maxsize=4)
        def _radial(shape, pixels, detail):
            base, fine, surface = _components(shape, pixels)
            candidate = base + detail * fine
            delta = gaussian_filter(candidate, (1, 1, 0), mode='reflect') - surface
            magnitude = np.sqrt(np.mean(delta ** 2, axis=2))
            mask = np.exp(-np.minimum(magnitude / 10.0, 10) ** 4)
            result = np.clip(np.rint(candidate - mask[..., None] * delta), 0, 255).astype(np.uint8)
            result.flags.writeable = False
            return result

        @lru_cache(maxsize=12)
        def _raw_fields(height, width, digest, seed, grid):
            rng = np.random.default_rng(int.from_bytes(digest, 'little') ^ seed & (1 << 64) - 1)
            independent = _smooth_field(rng, height, width, grid=grid, channels=3)
            shared = _smooth_field(rng, height, width, grid=grid, channels=1)
            return (independent, shared)

        @lru_cache(maxsize=12)
        def _mixed_field(height, width, digest, seed, grid, rho):
            independent, shared = _raw_fields(height, width, digest, seed, grid)
            reference = _raw_fields(height, width, digest, seed, 5)[0]
            if grid == 5 and rho == 0:
                result = independent.copy()
            else:
                mixed = math.sqrt(1 - rho) * independent.astype(np.float64) + math.sqrt(rho) * shared
                mixed -= mixed.mean(axis=(0, 1), keepdims=True)
                reference_rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2, axis=(0, 1)))
                mixed_rms = np.sqrt(np.mean(mixed ** 2, axis=(0, 1)))
                result = (mixed * (reference_rms / np.maximum(mixed_rms, 1e-08))).astype(np.float32)
            result.flags.writeable = False
            return result

        class RadialFieldAblation(AdaptiveLaunderingAttack):
            name = 'radial_field_ablation'

            def __init__(self, grid=5, rho=0.0, detail=0.6, amplitude=4.0, mode='RF', seed=42):
                super().__init__(seed=seed)
                if mode not in ('I', 'R', 'F', 'RF') or not isinstance(grid, int) or (not 2 <= grid <= 16):
                    raise ValueError('无效模式或网格尺寸')
                for value, low, high in ((rho, 0, 1), (detail, 0, 1), (amplitude, 0, 6)):
                    if not np.isfinite(value) or not low <= value <= high:
                        raise ValueError('相关性、细节或幅度无效')
                self.grid, self.rho, self.detail = (grid, float(rho), float(detail))
                self.amplitude, self.mode = (float(amplitude), mode)

            def field(self, image):
                digest = hashlib.blake2b(np.ascontiguousarray(image).tobytes(), digest_size=8).digest()
                return _mixed_field(*image.shape[:2], digest, self.seed, self.grid, self.rho)

            def _transform(self, image):
                if self.mode == 'I' or _flat_region_ratio(image) > 0.5:
                    return image.copy()
                pixels = np.ascontiguousarray(image).tobytes()
                base = _radial(image.shape, pixels, self.detail) if self.mode in ('R', 'RF') else image
                if self.mode == 'R':
                    return base.copy()
                result = base.astype(np.float32) + self.amplitude * self.field(image)
                return np.clip(np.rint(result), 0, 255).astype(np.uint8)

            def apply_with_reference(self, image, reference):
                """评测内部复用同一原图的保真参考；回退逐行对应冻结 V1。"""
                original = _validate_rgb(image)
                candidate = self._transform(original)
                similarity, psnr = reference.measure(candidate)
                alpha, checks = (1.0, 1)
                if similarity < self.min_ssim or psnr < self.min_psnr:
                    source = original.astype(np.float64)
                    delta = candidate.astype(np.float64) - source
                    best, alpha, similarity, psnr = (original.copy(), 0.0, 1.0, 99.0)
                    lo, hi = (0.0, 1.0)
                    for _ in range(self.search_steps):
                        mid = (lo + hi) / 2
                        blended = np.clip(np.rint(source + mid * delta), 0, 255).astype(np.uint8)
                        trial_ssim, trial_psnr = reference.measure(blended)
                        checks += 1
                        if trial_ssim >= self.min_ssim and trial_psnr >= self.min_psnr:
                            lo = mid
                            best, alpha = (blended, mid)
                            similarity, psnr = (trial_ssim, trial_psnr)
                        else:
                            hi = mid
                    candidate = best
                flat = _flat_region_ratio(candidate) > 0.5
                if flat:
                    candidate, alpha, similarity, psnr = (original.copy(), 0.0, 1.0, 99.0)
                return (np.ascontiguousarray(candidate), dict(alpha=alpha, ssim=similarity, psnr=psnr, fidelity_checks=checks, flat_region_fallback=flat))

            def apply_with_info(self, image):
                with threadpool_limits(limits=1):
                    image = _validate_rgb(image)
                    return self.apply_with_reference(image, _FidelityReference(image))

        return RadialFieldAblation(grid=5, rho=1.0, detail=0.60, amplitude=4.0, mode="RF", seed=seed)
