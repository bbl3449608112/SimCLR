# Mini-SimCLR 图像表征学习复现实验报告

## 1. 论文信息

- 论文名称：A Simple Framework for Contrastive Learning of Visual Representations
- 论文地址：https://arxiv.org/abs/2002.05709
- 官方代码参考：https://github.com/google-research/simclr

## 2. 任务说明

本实验复现的任务是自监督图像表征学习。

```text
预训练输入：无标签图像
预训练目标：让同一图像的两种增强视图在表征空间中更接近，让不同图像的表征更远
评估方式：冻结 encoder，训练 linear probe，报告 CIFAR-10 分类准确率
```

## 3. 数据集

- 数据集名称：CIFAR-10
- 数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
- 实际使用预训练图像数：
- 实际使用 linear probe 训练图像数：
- 实际使用测试图像数：
- 使用设备：CPU / GPU
- 总训练耗时：

## 4. 数据增强

请说明自己使用的增强策略：

| 增强方法 | 参数设置 |
|---|---|
| RandomResizedCrop |  |
| RandomHorizontalFlip |  |
| ColorJitter |  |
| RandomGrayscale |  |
| GaussianBlur |  |

请说明为什么这些增强适合 SimCLR：

```text
（在这里填写）
```

## 5. 模型结构

请说明自己的 Mini-SimCLR 结构：

```text
Image -> Two Augmented Views -> Shared Encoder -> Projection Head -> NT-Xent Loss
```

### 5.1 Encoder

- encoder 类型：ResNet-18 / 小型 CNN / 其他
- 输出特征维度：
- 是否使用预训练权重：

### 5.2 Projection Head

- MLP 层数：
- hidden dimension：
- output dimension：
- 是否使用 ReLU / BatchNorm：

### 5.3 Linear Probe

- encoder 是否冻结：
- linear classifier 输入维度：
- 类别数：

## 6. Loss 实现

请说明 NT-Xent loss 的实现方式：

- batch size：
- `2N` 个增强样本如何构造：
- 正样本索引如何确定：
- temperature：
- logits shape：

可以粘贴关键代码片段或伪代码。

## 7. 训练设置

### 7.1 自监督预训练

| 配置 | 数值 |
|---|---:|
| train images |  |
| epochs |  |
| batch size |  |
| optimizer |  |
| learning rate |  |
| temperature |  |
| encoder | 小型 CNN / ResNet-18 / 其他 |
| device | CPU / GPU |

### 7.2 Linear Probe

| 配置 | 数值 |
|---|---:|
| train images |  |
| test images |  |
| epochs |  |
| batch size |  |
| optimizer |  |
| learning rate |  |
| device | CPU / GPU |

## 8. 训练过程

粘贴 contrastive loss 日志或 loss 曲线。

示例：

| Epoch | Contrastive Loss |
|---|---:|
| 1 |  |
| 2 |  |
| 3 |  |

请简要描述 loss 是否下降，以及训练是否稳定：

```text
（在这里填写）
```

## 9. Linear Probe 结果

| 指标 | 结果 |
|---|---:|
| test accuracy |  |
| random baseline | 10% |

请分析结果是否明显高于随机猜测：

```text
（在这里填写）
```

## 10. 预测结果展示

至少展示 3 个测试样例。

| 图片编号 | 真实类别 | 预测类别 | 是否正确 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题；
- 最终如何解决；
- 如果继续改进，可以从哪些方面入手，例如 batch size、epoch、temperature、projection head、数据增强等。

```text
（在这里填写）
```

## 12. AI 对话过程记录

- 录制工具：
- 对话链接：
- 使用的 AI 模型：
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节提供帮助，以及哪些部分是自己独立完成或验证的：

```text
（在这里填写）
```

## 13. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```
