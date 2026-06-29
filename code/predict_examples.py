from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from augmentations import CIFAR10_MEAN, CIFAR10_STD
from data import CIFAR10_CLASSES, DatasetConfig, build_prediction_dataset
from model import LinearProbe, SmallCNNEncoder
from utils import ensure_dir, get_device, load_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save CIFAR-10 prediction examples for the report.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pretrain-checkpoint", default="checkpoints/simclr_encoder.pt")
    parser.add_argument("--linear-checkpoint", default="checkpoints/linear_probe.pt")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", default="report/figures")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def unnormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cpu=args.cpu)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(args.results_dir)

    config = DatasetConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        train_limit=None,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    dataset = build_prediction_dataset(config)
    loader = DataLoader(dataset, batch_size=args.num_examples, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))

    encoder = SmallCNNEncoder(feature_dim=args.feature_dim)
    pretrain_checkpoint = load_checkpoint(args.pretrain_checkpoint, device)
    encoder.load_state_dict(pretrain_checkpoint["encoder_state_dict"])
    model = LinearProbe(encoder, feature_dim=args.feature_dim, num_classes=10).to(device)
    linear_checkpoint = load_checkpoint(args.linear_checkpoint, device)
    model.load_state_dict(linear_checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()

    examples = []
    cols = min(args.num_examples, len(images))
    plt.figure(figsize=(2.2 * cols, 2.8))
    for index in range(cols):
        true_label = CIFAR10_CLASSES[int(labels[index])]
        pred_label = CIFAR10_CLASSES[int(predictions[index])]
        correct = bool(labels[index] == predictions[index])
        examples.append(
            {
                "index": index,
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": correct,
            }
        )
        ax = plt.subplot(1, cols, index + 1)
        image = unnormalize(images[index]).permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    figure_path = output_dir / "prediction_examples.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()
    save_json({"examples": examples}, Path(args.results_dir) / "prediction_examples.json")
    print(f"saved {figure_path}")
    print(json.dumps(examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
