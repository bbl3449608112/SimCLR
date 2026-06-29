from __future__ import annotations

import ast
from pathlib import Path


REQUIRED_FILES = [
    "requirements.txt",
    "README.md",
    "code/augmentations.py",
    "code/data.py",
    "code/model.py",
    "code/losses.py",
    "code/train_simclr.py",
    "code/linear_probe.py",
    "code/visualize_results.py",
    "code/predict_examples.py",
    "report/report.md",
]

REQUIRED_SNIPPETS = {
    "code/augmentations.py": [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "ColorJitter",
        "RandomGrayscale",
    ],
    "code/losses.py": ["nt_xent_loss", "temperature", "cross_entropy"],
    "code/model.py": ["SmallCNNEncoder", "ProjectionHead", "SimCLR", "LinearProbe"],
    "code/train_simclr.py": ["build_pretrain_dataset", "nt_xent_loss", "torch.save"],
    "code/linear_probe.py": ["encoder.load_state_dict", "requires_grad", "test_accuracy"],
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    for path in sorted((root / "code").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for relative_path, snippets in REQUIRED_SNIPPETS.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        absent = [snippet for snippet in snippets if snippet not in text]
        if absent:
            raise SystemExit(f"{relative_path} missing expected snippets: {absent}")

    print("structure validation passed")


if __name__ == "__main__":
    main()
