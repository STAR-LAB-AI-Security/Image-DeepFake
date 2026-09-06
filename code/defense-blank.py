class Defense:
    """DeepFake 检测器（学生自选算法，不指定特定防御方法）。

    固定接口（test.py 调用契约）：
        __init__(C=1.0, max_iter=500, seed=42)
            # test.py 调用 Defense() 与 build_defense("frequency_stat")
            # -> Defense(C=1.0, max_iter=500, seed=42)，三个关键字必须能接收（或 **kwargs 兜底）
        fit(images, labels) -> self          # 在 (真实+DeepFake) 训练集上拟合
        predict(image) -> float              # 返回 P(fake|image) ∈ [0,1]，越大越像伪造

    提示：backend 保留 FeatureDetector 基类（StandardScaler + LogisticRegression 的
    fit/predict 机械）与 extract_frequency_features / extract_pixel_features 特征工具；
    可继承 FeatureDetector 只实现 _extract(image) -> feature_vector，也可完全自定义。
    """

    name = "defense"

    def __init__(self, C: float = 1.0, max_iter: int = 500, seed: int = 42, **kwargs):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, images, labels):
        pass

    def predict(self, image) -> float:
        pass
