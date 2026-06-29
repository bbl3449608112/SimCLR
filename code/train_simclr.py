from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DatasetConfig, build_pretrain_dataset
from losses import nt_xent_loss
from model import SimCLR
from utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a Mini-SimCLR model on CIFAR-10.")
    parser.add_argument("--data-dir", default="data", help="Directory where CIFAR-10 is stored/downloaded.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of self-supervised pretraining epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for contrastive pretraining.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature.")
    parser.add_argument("--train-limit", type=int, default=2000, help="Number of CIFAR-10 train images to use.")
    parser.add_argument("--image-size", type=int, default=32, help="Input image size.")
    parser.add_argument("--feature-dim", type=int, default=256, help="Encoder output dimension.")
    parser.add_argument("--projection-dim", type=int, default=128, help="Projection head output dimension.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is safest on Windows/CPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for model checkpoints.")
    parser.add_argument("--results-dir", default="results", help="Directory for JSON summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cpu=args.cpu)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.results_dir)
    ensure_dir("logs")

    config = DatasetConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        train_limit=args.train_limit,
        test_limit=None,
        seed=args.seed,
    )
    dataset = build_pretrain_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SimCLR(feature_dim=args.feature_dim, projection_dim=args.projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = {
        "config": vars(args),
        "device": str(device),
        "num_train_images": len(dataset),
        "epochs": [],
    }
    log_path = Path("logs") / "pretrain.log"
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = 0
        progress = tqdm(loader, desc=f"pretrain epoch {epoch}/{args.epochs}", leave=False)
        for view_1, view_2 in progress:
            view_1 = view_1.to(device, non_blocking=True)
            view_2 = view_2.to(device, non_blocking=True)

            _, z1 = model(view_1)
            _, z2 = model(view_2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / max(total_batches, 1)
        epoch_record = {
            "epoch": epoch,
            "contrastive_loss": avg_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history["epochs"].append(epoch_record)
        line = (
            f"epoch={epoch:03d} "
            f"contrastive_loss={avg_loss:.6f} "
            f"lr={epoch_record['learning_rate']:.6g}\n"
        )
        print(line.strip())
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    history["elapsed_seconds"] = round(time.time() - started_at, 2)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "config": vars(args),
        "history": history,
    }
    torch.save(checkpoint, Path(args.checkpoint_dir) / "simclr_encoder.pt")
    save_json(history, Path(args.results_dir) / "pretrain_history.json")
    print(f"saved checkpoint to {Path(args.checkpoint_dir) / 'simclr_encoder.pt'}")
    print(f"saved history to {Path(args.results_dir) / 'pretrain_history.json'}")


if __name__ == "__main__":
    main()
