# Mini-SimCLR 图像表征学习复现实验报告

## 1. 论文信息

- 论文名称：A Simple Framework for Contrastive Learning of Visual Representations
- 论文地址：https://arxiv.org/abs/2002.05709
- 官方代码参考：https://github.com/google-research/simclr

本项目复现论文中的核心思想：同一张图像经过两次随机增强得到正样本对，不同图像的增强结果作为负样本，通过对比学习训练 encoder，再冻结 encoder 用 linear probe 评估表征质量。

## 2. 任务说明

预训练阶段输入无标签 CIFAR-10 图像，只使用图像本身产生两种增强视图。评估阶段丢弃 projection head，冻结 encoder，在 encoder 输出后接线性分类器，用 CIFAR-10 标签训练和测试分类准确率。

## 3. 数据集

- 数据集：CIFAR-10
- 下载方式：`torchvision.datasets.CIFAR10(download=True)`
- 预训练默认图像数：2000
- Linear probe 默认训练图像数：2000
- Linear probe 默认测试图像数：1000
- 自监督预训练阶段：不使用标签
- 线性评估阶段：只训练线性分类器

本地 Codex 验证环境没有安装 `torch/torchvision`，因此本报告先给出完整实现、命令和结果记录位置；真实数值需要在 PyTorch 环境运行脚本后由 `results/*.json` 填入。

## 4. 数据增强

| 增强方法 | 参数设置 |
|---|---|
| RandomResizedCrop | size=32, scale=(0.2, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| ColorJitter | brightness/contrast/saturation=0.4, hue=0.1, p=0.8 |
| RandomGrayscale | p=0.2 |
| GaussianBlur | kernel_size=3, sigma=(0.1, 2.0) |
| Normalize | CIFAR-10 mean/std |

这些增强可以让同一图像的两个视图在颜色、局部区域和轻微模糊上不同，迫使 encoder 学习更稳定的语义表征，而不是记忆像素级细节。

## 5. 模型结构

```text
Image -> Two Augmented Views -> Shared CNN Encoder -> Projection Head -> NT-Xent Loss
```

### 5.1 Encoder

- 类型：小型 CNN encoder
- 结构：4 个卷积块，每个卷积块包含 Conv-BN-ReLU-Conv-BN-ReLU
- 下采样：后 3 个卷积块 stride=2
- pooling：Global Average Pooling
- 输出特征维度：256
- 预训练权重：不使用外部预训练权重

### 5.2 Projection Head

- 结构：Linear -> BatchNorm1d -> ReLU -> Linear
- 输入维度：256
- hidden dimension：512
- 输出维度：128
- 输出后进行 L2 normalize

### 5.3 Linear Probe

- 冻结 encoder：是
- 分类器：单层 `nn.Linear(256, 10)`
- 类别数：10
- 训练目标：cross entropy

## 6. Loss 实现

NT-Xent loss 位于 `code/losses.py`，核心步骤如下：

1. 输入 `z1` 和 `z2`，形状均为 `[N, D]`；
2. 拼接为 `[2N, D]`；
3. 对所有 projection 做 L2 normalize；
4. 计算 `[2N, 2N]` 的相似度矩阵；
5. 将对角线自身相似度 mask 为 `-inf`；
6. 第 `i` 个样本的正样本索引是 `(i + N) % (2N)`；
7. 对 temperature-scaled logits 使用 `cross_entropy`。

默认 temperature 为 0.5。

## 7. 训练设置

### 7.1 自监督预训练

| 配置 | 数值 |
|---|---:|
| train images | 2000 |
| epochs | 5 |
| batch size | 64 |
| optimizer | Adam |
| learning rate | 1e-3 |
| weight decay | 1e-6 |
| temperature | 0.5 |
| encoder | 小型 CNN |

运行命令：

```bash
python code/train_simclr.py --train-limit 2000 --epochs 5 --batch-size 64 --cpu
```

### 7.2 Linear Probe

| 配置 | 数值 |
|---|---:|
| train images | 2000 |
| test images | 1000 |
| epochs | 3 |
| batch size | 128 |
| optimizer | Adam |
| learning rate | 1e-3 |
| classifier | Linear(256, 10) |

运行命令：

```bash
python code/linear_probe.py --train-limit 2000 --test-limit 1000 --epochs 3 --batch-size 128 --cpu
```

## 8. 训练过程记录

预训练脚本会输出：

- `logs/pretrain.log`
- `results/pretrain_history.json`
- `report/figures/pretrain_loss.png`

运行后在此处填入 loss 表：

| Epoch | Contrastive Loss |
|---|---:|
| 1 | 4.2626 |
| 2 | 4.1050 |
| 3 | 3.9836 |
| 4 | 3.9070 |
| 5 | 3.8761 |

loss 曲线文件：`report/figures/pretrain_loss.png`

## 9. Linear Probe 结果

评估脚本会输出：

- `logs/linear_probe.log`
- `results/linear_probe_results.json`
- `report/figures/linear_probe_accuracy.png`

| 指标 | 结果 |
|---|---:|
| test accuracy | 35.50% |
| random baseline | 10% |

如果准确率接近 10%，可能原因包括：预训练 epoch 太少、batch size 太小、CPU 版本 encoder 较浅、训练图像数量较少、增强较强或 temperature 需要调参。

## 10. 预测结果展示

预测展示脚本：

```bash
python code/predict_examples.py --num-examples 5 --cpu
```

输出文件：

- `report/figures/prediction_examples.png`
- `results/prediction_examples.json`

运行后在此处填入至少 5 个样例：

| 图片编号 | 真实类别 | 预测类别 | 是否正确 |
|---|---|---|---|
| 1 | cat | dog | 否 |
| 2 | automobile | automobile | 是 |
| 3 | horse | bird | 否 |
| 4 | ship | horse | 否 |
| 5 | ship | ship | 是 |

## 11. 问题与改进

本项目优先满足 CPU 友好和流程完整，因此使用小型 CNN 与较少训练图像。后续可改进方向：

- 增加预训练 epoch；
- 使用更大 batch size；
- 使用 ResNet-18 作为 encoder；
- 对比不同 temperature；
- 增加 train_limit 到 5000 或完整 CIFAR-10；
- 加入 t-SNE/nearest neighbor retrieval 可视化。

## 12. AI 对话过程记录

- 使用工具：Codex / ChatGPT
- 对话链接：提交前请填写可访问链接
- 主要帮助环节：理解题目要求、拆分模块、实现双视图增强、模型、NT-Xent、训练脚本、评估脚本、报告结构和本地验证。

## 13. Git 提交记录

提交前可运行：

```bash
git log --oneline
```

当前本地仓库已经按模块做了小步 commit；请在正式提交时粘贴输出。
