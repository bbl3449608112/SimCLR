# Mini-SimCLR 图像表征学习复现

这是一个面向 CIFAR-10 小子集的 CPU 友好版 SimCLR 复现项目。实现内容包括：

- CIFAR-10 数据读取与双视图随机增强；
- 小型 CNN encoder；
- MLP projection head；
- 手写 NT-Xent contrastive loss；
- 自监督预训练脚本；
- 冻结 encoder 的 linear probe 评估脚本；
- loss/accuracy 曲线与测试样例预测图生成脚本；
- 实验报告草稿与本地结构检查脚本。

## 环境安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果使用 CPU，默认参数即可。若有 GPU，脚本会自动使用 CUDA；也可以加 `--cpu` 强制使用 CPU。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── augmentations.py
│   ├── data.py
│   ├── losses.py
│   ├── model.py
│   ├── train_simclr.py
│   ├── linear_probe.py
│   ├── visualize_results.py
│   ├── predict_examples.py
│   └── validate_structure.py
├── checkpoints/
├── data/
├── logs/
├── report/
│   ├── report.md
│   └── figures/
└── results/
```

`data/` 和 `checkpoints/` 不提交到 Git。CIFAR-10 会由 `torchvision.datasets.CIFAR10` 自动下载。

## 训练流程

### 1. 自监督预训练

CPU 友好配置：

```bash
python code/train_simclr.py --train-limit 2000 --epochs 5 --batch-size 64 --cpu
```

更快的 smoke test：

```bash
python code/train_simclr.py --train-limit 256 --epochs 1 --batch-size 32 --cpu
```

输出：

- `checkpoints/simclr_encoder.pt`
- `results/pretrain_history.json`
- `logs/pretrain.log`

### 2. Linear probe

```bash
python code/linear_probe.py --train-limit 2000 --test-limit 1000 --epochs 3 --batch-size 128 --cpu
```

输出：

- `checkpoints/linear_probe.pt`
- `results/linear_probe_results.json`
- `logs/linear_probe.log`

### 3. 生成曲线

```bash
python code/visualize_results.py
```

输出：

- `report/figures/pretrain_loss.png`
- `report/figures/linear_probe_accuracy.png`

### 4. 生成预测样例图

```bash
python code/predict_examples.py --num-examples 5 --cpu
```

输出：

- `report/figures/prediction_examples.png`
- `results/prediction_examples.json`

## 实现要点

### 双视图增强

`code/augmentations.py` 使用以下增强组成 SimCLR pipeline：

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomGrayscale`
- `GaussianBlur`
- `Normalize`

`TwoViewDataset` 对同一张原图连续调用两次随机 transform，返回 `view_1` 和 `view_2`，预训练阶段不使用标签。

### 模型

模型位于 `code/model.py`：

- `SmallCNNEncoder`: 4 个卷积块 + global average pooling，输出 256 维特征；
- `ProjectionHead`: `Linear -> BatchNorm -> ReLU -> Linear`，输出 128 维 projection；
- `SimCLR`: 共享 encoder，对两种增强视图提取表征；
- `LinearProbe`: 冻结 encoder，只训练线性分类器。

### NT-Xent loss

`code/losses.py` 手写实现 NT-Xent：

1. 拼接 `z1`、`z2` 得到 `2N` 个 projection；
2. 对 projection 做 L2 normalize；
3. 计算 `2N x 2N` cosine similarity；
4. mask 掉自身相似度；
5. 正样本索引为 `(i + N) % (2N)`；
6. 使用 temperature-scaled logits 和 cross entropy 计算 loss。

## 本地检查

当前 Codex 环境没有安装 `torch/torchvision`，因此这里提供了不依赖 PyTorch 的结构和语法检查：

```bash
python code/validate_structure.py
```

正式提交前建议在安装好 PyTorch 的环境中完整跑一遍：

```bash
python code/train_simclr.py --train-limit 2000 --epochs 5 --batch-size 64
python code/linear_probe.py --train-limit 2000 --test-limit 1000 --epochs 3 --batch-size 128
python code/visualize_results.py
python code/predict_examples.py --num-examples 5
```

然后把 `results/*.json` 中的最终数值和 `report/figures/*.png` 补到 `report/report.md`。

## 提交说明

题目要求包含 AI 对话记录和 git 小步提交记录。当前仓库已按模块进行了本地 commit，可用：

```bash
git log --oneline
```

查看记录。推送到你的 GitHub/Gitee 仓库前，请把本地 git 作者信息改成自己的姓名和邮箱。
