class JPEGReencodeAttack:
    """痕迹隐匿攻击（学生自选算法，无需与 baseline 相同）。

    固定接口（test.py 调用契约）：
        __init__(quality=85, seed=42)      # 必须能接收 test.py 传入的 quality、seed
        apply(image) -> np.ndarray         # 同尺寸 RGB uint8（隐匿后图像）

    模块级 attack(sample, quality=85, seed=42) 委托到本类；backend 保留
    to_pil / from_pil / resize_pair 等工具可用。
    """

    name = "jpeg"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.quality = quality
        self.seed = seed

    def apply(self, image: np.ndarray) -> np.ndarray:
        """痕迹隐匿（学生自选算法）。

        参数
        ----
        image : RGB uint8 H×W×3（256×256 测试伪造图）

        返回
        ----
        同尺寸、同 dtype 的 np.ndarray（隐匿后图像）
        """
        pass
