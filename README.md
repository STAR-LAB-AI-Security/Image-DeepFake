# 图像痕迹隐匿与鲁棒检测 —— 代码说明（README）

围绕统一的 DeepFake 图像检测服务 **DeepGuard-Image** 的"伪造痕迹隐匿—鲁棒检测"攻防基准：攻击方对主办方提供的 DeepFake 图像施加**轻量后处理**（JPEG 重编码等），在保持视觉质量基本不变的前提下降低未知检测器对伪造图像的识别能力；防御方提交仅凭图像输出 DeepFake 概率的检测算法，需同时应对未处理 DeepFake、公开后处理攻击与隐藏组合攻击。

> ⚠️ **轻量化说明**：真实 DeepfakeBench（Xception/MesoNet 等大 CNN）在 CPU 上不可行，本基准交付**忠实于赛题概念的轻量化实现**——内置确定性"合成人脸图像集"（真实图 + 注入融合边界痕迹的伪造图），检测器为频域/像素统计特征 + 逻辑回归，攻击为 PIL/numpy 后处理，全部 CPU 秒级~分钟级跑通，结果可复现。

## 文件职责

| 文件 | 职责 |
|---|---|
| `dataset.py` | 基准数据集：生成/缓存 256×256 合成人脸图像集（真/假各半，含 sample_id），确定性划分（训练 192 / 测试 96，seed 42），无网可跑 |
| `attack.py` | 攻击代码：交付 Baseline **JPEGReencodeAttack**（JPEG 重编码, quality=85）＋ 赛题固定攻击池 A0（identity/jpeg/resize_recover/blur_sharpen/color_laundering/frequency_laundering/composite_hidden） |
| `defense.py` | 防御代码：交付 Baseline **FrequencyStatDefense**（SRM 噪声残差 + 分块 DCT + FFT 频带 + 颜色统计特征 + 逻辑回归，31 维）＋ 固定防御池 D0 成员 **PixelStatDefense**（15 维空域统计） |
| `test.py` | 评测入口：指标计算（AUC/ACC/EER/ASR/SSIM/PSNR/RobustAUC/资格线）；`--stage attack\|defense\|all` |
| `run_attack.sh` / `run_defense.sh` | 终端脚本：`bash run_attack.sh`（攻击评测）/ `bash run_defense.sh`（防御评测） |
| `requirements.txt` | 最小依赖：numpy / scikit-learn / scipy / Pillow |

## 快速开始

```bash
pip install -r requirements.txt
bash run_attack.sh      # 攻击方 Baseline 评测（JPEG 隐匿 vs 固定防御池 D0）
bash run_defense.sh     # 防守方 Baseline 评测（FrequencyStat vs 固定攻击池 A0）
```

也可直接运行（等价）：

```bash
python test.py --stage attack
python test.py --stage defense
python test.py --stage all        # 完整攻防评测
```

结果保存在 `results/metrics.json`。

## 评测协议（与题目文档一致）

- **数据**：合成人脸图像集，训练 192 张（96 真 + 96 假）、测试 96 张（48 真 + 48 假），256×256×3 RGB uint8，标签 {0=real, 1=fake}；测试集全程干净、不参与训练（隐藏测试集），seed=42 固定划分。
- **攻击接口**（赛题 §5.4）：`attack(sample) -> {"image": RGB uint8}`，输出与输入同尺寸同 dtype；交付 Baseline 为 `JPEGReencodeAttack(quality=85)`。
- **防御接口**（赛题 §7.4）：`defend(request) -> {"fake_probability": float ∈ [0,1]}`；交付 Baseline 为 `FrequencyStatDefense`（特征 + StandardScaler + LogisticRegression，C=1.0, max_iter=500）。
- **固定阈值**：τ=0.5；视觉保真软约束 SSIM ≥ 0.92、PSNR ≥ 28 dB；攻击资格线 = 合法攻击率 ≥ 0.80 且平均 SSIM ≥ 0.92。
- **攻击方得分**：交付攻击在各防御池检测器上的 ASR 均值；**防御方得分**：(CleanAUC + RobustAUC) / 2。

## 指标公式

$$Accuracy(\tau{=}0.5) = \frac{\#\{p_f \ge 0.5 \text{ 且真实为假}\} + \#\{p_f < 0.5 \text{ 且真实为真}\}}{N}$$

$$ASR(a,d) = \frac{1}{N_{\text{fake}}} \sum_{i=1}^{N_{\text{fake}}} \mathbb{1}\big[valid_i = 1 \;\land\; p_{f,i} < 0.5\big]$$

$$SSIM(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}, \qquad
PSNR(x,y) = 10\log_{10}\frac{255^2}{MSE(x,y)}$$

$$RobustAUC = AUC\big(\{p_f(I'_j)\}_{j=1}^{M}, \{p_f(I_k)\}_{k=1}^{N_{\text{real}}}\big)$$

其中 $I'_j$ 为全部合法攻击后的伪造图，$I_k$ 为原始真实图，$M$ 为合法攻击后伪造图总数。

## 参考实测结果（默认参数，见 results/metrics.json）

数值随硬件/运行略有波动：

| 阶段 | 关键指标 | 实测值 |
|---|---|---|
| 攻击 | CleanAUC（frequency_stat / pixel_stat） | 1.0000 / 1.0000 |
| 攻击 | ASR（frequency_stat / pixel_stat） | 1.0000 / 0.8333 |
| 攻击 | 平均 ASR（跨检测器） | 0.9167 |
| 攻击 | 平均 SSIM / PSNR | ~0.89 / ~38 dB（SSIM 未达 0.92 资格线，与主办方后端对 JPEG 的实测一致） |
| 防御 | CleanAUC / CleanACC / CleanEER | ~1.0 / ~1.0 / ~0.0 |
| 防御 | RobustAUC / 防御得分 | ~0.29 / ~0.64 |
| 防御 | 隐藏组合攻击 AUC（composite_hidden） | 0.0 |

> 说明：频域统计检测器对合成痕迹的决策高度依赖高频分量，重编码/滤波类攻击将其完全移除（攻击后伪造图分数饱和到真实图之下），因此 RobustAUC 显著低于 CleanAUC——这正是"单特征家族检测器鲁棒性差、防御方需多特征家族与攻击增强训练"这一赛题要点的量化体现；颜色洗白攻击对其基本无效（AUC≈1.0）。
