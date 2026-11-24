"""模型服务：单作业内存状态、训练/预测/扩列/保存/重载。

进程内只保留一个活动模型（单作业）。训练、扩列、保存、重载
通过同一把异步锁串行化；预测为只读操作，不与锁互斥
（PyTorch 张量读取与训练更新在 GIL 下对本作业足够安全，
且扩列/保存会整体替换 bundle 引用）。
"""
from __future__ import annotations

import asyncio
import os

import torch

from .checkpoint import (
    CheckpointBundle,
    expand_bundle,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .data import align_labels, align_rows


class ModelService:
    def __init__(self, bundle: CheckpointBundle):
        self._bundle = bundle
        self._lock = asyncio.Lock()

    # ---- 基础信息 -------------------------------------------------
    @property
    def bundle(self) -> CheckpointBundle:
        return self._bundle

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def info(self) -> dict:
        b = self._bundle
        return {
            "feature_names": list(b.feature_names),
            "input_dim": len(b.feature_names),
            "hidden_dim": b.hidden_dim,
            "num_classes": b.num_classes,
            "dtype": str(b.dtype).replace("torch.", ""),
            "optimizer_step": _read_step(b.optimizer),
            "source_path": b.source_path,
        }

    # ---- 预测 -----------------------------------------------------
    @torch.no_grad()
    def predict(self, instances: list[dict]) -> list[dict]:
        b = self._bundle
        x, missing = align_rows(instances, b.feature_names, b.dtype)
        logits = b.model(x)
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        return [
            {
                "label": int(labels[i]),
                "score": float(probs[i, labels[i]]),
                "probabilities": [float(p) for p in probs[i]],
                "zero_filled_columns": missing[i],
            }
            for i in range(len(instances))
        ]

    # ---- 训练一步 -------------------------------------------------
    def train_step(self, instances: list[dict], labels: list[int]) -> dict:
        if len(instances) != len(labels):
            raise ValueError(
                f"instances 数量({len(instances)})与 labels 数量({len(labels)})不一致"
            )
        b = self._bundle
        x, _ = align_rows(instances, b.feature_names, b.dtype)
        y = align_labels(labels, b.num_classes)

        criterion = torch.nn.CrossEntropyLoss()
        b.model.train()

        with torch.no_grad():
            loss_before = float(criterion(b.model(x), y))

        b.optimizer.zero_grad()
        logits = b.model(x)
        loss = criterion(logits, y)
        loss.backward()
        b.optimizer.step()

        with torch.no_grad():
            loss_after = float(criterion(b.model(x), y))

        return {
            "step": _read_step(b.optimizer),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "num_samples": len(instances),
            "feature_names": list(b.feature_names),
        }

    # ---- 扩列 -----------------------------------------------------
    def expand(self, new_feature_names: list[str]) -> dict:
        names = list(dict.fromkeys(new_feature_names))  # 去重但保序
        if len(names) != len(new_feature_names):
            raise ValueError("new_feature_names 内部存在重复列名")
        new_bundle = expand_bundle(self._bundle, names)
        self._bundle = new_bundle  # 整体替换，旧引用若正在预测仍可安全读完
        return {
            "feature_names": list(new_bundle.feature_names),
            "input_dim": len(new_bundle.feature_names),
            "added": names,
        }

    # ---- 保存（原子） ---------------------------------------------
    def save(self, path: str | None) -> str:
        target = os.path.abspath(path) if path else self._bundle.source_path
        save_checkpoint_atomic(self._bundle, target)
        return target

    # ---- 从磁盘重载（重启/回退用） --------------------------------
    def reload(self, path: str | None = None) -> dict:
        target = os.path.abspath(path) if path else self._bundle.source_path
        b = load_checkpoint(target, dtype=self._bundle.dtype)
        self._bundle = b
        return {
            "path": b.source_path,
            "feature_names": list(b.feature_names),
            "input_dim": len(b.feature_names),
            "optimizer_step": _read_step(b.optimizer),
        }


def _read_step(optimizer: torch.optim.Optimizer) -> int:
    """兼容不同 PyTorch 版本读取 Adam 的 step。"""
    state = optimizer.state
    if not state:
        return 0
    step = next(iter(state.values())).get("step", 0)
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)
