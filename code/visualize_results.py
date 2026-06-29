from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from utils import ensure_dir


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Mini-SimCLR training curves.")
    parser.add_argument("--pretrain-json", default="results/pretrain_history.json")
    parser.add_argument("--linear-json", default="results/linear_probe_results.json")
    parser.add_argument("--output-dir", default="report/figures")
    return parser.parse_args()


def plot_pretrain_loss(history: dict, output_dir: Path) -> None:
    epochs = [item["epoch"] for item in history.get("epochs", [])]
    losses = [item["contrastive_loss"] for item in history.get("epochs", [])]
    if not epochs:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o", color="#2563eb")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive loss")
    plt.title("Mini-SimCLR Pretraining Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pretrain_loss.png", dpi=160)
    plt.close()


def plot_linear_probe(history: dict, output_dir: Path) -> None:
    epochs = [item["epoch"] for item in history.get("epochs", [])]
    train_acc = [item["train_accuracy"] for item in history.get("epochs", [])]
    test_acc = [item["test_accuracy"] for item in history.get("epochs", [])]
    if not epochs:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc, marker="o", label="train", color="#16a34a")
    plt.plot(epochs, test_acc, marker="s", label="test", color="#dc2626")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Linear Probe Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "linear_probe_accuracy.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    pretrain_path = Path(args.pretrain_json)
    if pretrain_path.exists():
        plot_pretrain_loss(_load_json(pretrain_path), output_dir)
        print(f"saved {output_dir / 'pretrain_loss.png'}")
    else:
        print(f"skip pretrain plot; missing {pretrain_path}")

    linear_path = Path(args.linear_json)
    if linear_path.exists():
        plot_linear_probe(_load_json(linear_path), output_dir)
        print(f"saved {output_dir / 'linear_probe_accuracy.png'}")
    else:
        print(f"skip linear plot; missing {linear_path}")


if __name__ == "__main__":
    main()
