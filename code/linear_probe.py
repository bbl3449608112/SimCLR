from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DatasetConfig, build_linear_probe_datasets
from model import LinearProbe, SmallCNNEncoder
from utils import ensure_dir, get_device, load_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a frozen linear probe.")
    parser.add_argument("--data-dir", default="data", help="Directory where CIFAR-10 is stored/downloaded.")
    parser.add_argument("--checkpoint", default="checkpoints/simclr_encoder.pt", help="Pretrained checkpoint path.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of linear probe epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for linear probing.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for classifier.")
    parser.add_argument("--train-limit", type=int, default=2000, help="Number of labeled train images.")
    parser.add_argument("--test-limit", type=int, default=1000, help="Number of labeled test images.")
    parser.add_argument("--image-size", type=int, default=32, help="Input image size.")
    parser.add_argument("--feature-dim", type=int, default=256, help="Encoder feature dimension.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--results-dir", default="results", help="Directory for JSON summaries.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for linear-probe checkpoint.")
    return parser.parse_args()


def evaluate(model: LinearProbe, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cpu=args.cpu)
    ensure_dir(args.results_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir("logs")

    config = DatasetConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    train_dataset, test_dataset = build_linear_probe_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    encoder = SmallCNNEncoder(feature_dim=args.feature_dim)
    checkpoint = load_checkpoint(args.checkpoint, device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model = LinearProbe(encoder, feature_dim=args.feature_dim, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-6)

    history = {
        "config": vars(args),
        "device": str(device),
        "num_train_images": len(train_dataset),
        "num_test_images": len(test_dataset),
        "epochs": [],
    }
    log_path = Path("logs") / "linear_probe.log"
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"linear epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history["epochs"].append(epoch_record)
        line = (
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"train_acc={train_accuracy:.4f} "
            f"test_loss={test_loss:.6f} "
            f"test_acc={test_accuracy:.4f}\n"
        )
        print(line.strip())
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    history["elapsed_seconds"] = round(time.time() - started_at, 2)
    final_epoch = history["epochs"][-1] if history["epochs"] else {}
    history["final_test_accuracy"] = final_epoch.get("test_accuracy")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "config": vars(args),
            "history": history,
        },
        Path(args.checkpoint_dir) / "linear_probe.pt",
    )
    save_json(history, Path(args.results_dir) / "linear_probe_results.json")
    print(f"final test accuracy: {history['final_test_accuracy']:.4f}")
    print(f"saved results to {Path(args.results_dir) / 'linear_probe_results.json'}")


if __name__ == "__main__":
    main()
