"""生成初始检查点。

构造一个 4 个原始测点字段的三分类合成数据集，训练一个
两层分类模型（4 -> 8 -> 3），并把模型权重与 Adam 状态
一起保存到 checkpoints/model.pt。

用法：
    python scripts/train_initial.py [--output checkpoints/model.pt] [--steps 200]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from app.checkpoint import CHECKPOINT_VERSION
from app.model import TwoLayerClassifier

FEATURE_NAMES = ["temp", "pressure", "rpm", "current"]
HIDDEN_DIM = 8
NUM_CLASSES = 3
SEED = 20261124


def make_dataset(n_per_class: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """三个类别的高斯簇合成数据，列顺序与 FEATURE_NAMES 对齐。"""
    g = torch.Generator().manual_seed(SEED)
    centers = torch.tensor(
        [
            [1.0, -1.0, 0.5, 0.0],
            [-1.0, 1.0, -0.5, 0.8],
            [0.5, 0.5, 1.5, -1.2],
        ],
        dtype=torch.float32,
    )
    xs, ys = [], []
    for label in range(NUM_CLASSES):
        noise = torch.randn(n_per_class, len(FEATURE_NAMES), generator=g) * 0.6
        xs.append(centers[label] + noise)
        ys.append(torch.full((n_per_class,), label, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/model.pt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    x, y = make_dataset()
    model = TwoLayerClassifier(len(FEATURE_NAMES), HIDDEN_DIM, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    step = 0
    for epoch in range(1000):
        g = torch.Generator().manual_seed(args.seed + epoch)
        perm = torch.randperm(x.shape[0], generator=g)
        for start in range(0, x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            step += 1
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(dim=1) == y).float().mean().item()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    blob = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_names": list(FEATURE_NAMES),
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "dtype": "float32",
        "version": CHECKPOINT_VERSION,
    }
    torch.save(blob, args.output)
    print(f"已保存初始检查点: {os.path.abspath(args.output)}")
    print(f"训练步数: {step}，训练集准确率: {acc:.4f}")
    print(f"特征列: {FEATURE_NAMES}")


if __name__ == "__main__":
    main()
