"""检查点（含 Adam 状态）的加载、扩列与原子保存。

检查点文件格式（torch.save 的 zip 归档）：
{
    "model_state":      模型 state_dict,
    "optimizer_state":  Adam optimizer.state_dict(),
    "feature_names":    当前模型输入列名（有序）,
    "hidden_dim":       隐藏层宽度,
    "num_classes":      类别数,
    "dtype":            浮点精度字符串,
    "version":          检查点格式版本,
}
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import torch

from .model import TwoLayerClassifier, expand_model

CHECKPOINT_VERSION = 1


@dataclass
class CheckpointBundle:
    """内存中活动的模型 + 优化器 + 元数据。"""

    model: TwoLayerClassifier
    optimizer: torch.optim.Optimizer
    feature_names: list[str]
    hidden_dim: int
    num_classes: int
    dtype: torch.dtype
    source_path: str

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "feature_names": list(self.feature_names),
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dtype": str(self.dtype).replace("torch.", ""),
            "version": CHECKPOINT_VERSION,
        }


_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _resolve_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    if name not in _DTYPE_MAP:
        raise ValueError(f"不支持的 dtype: {name}")
    return _DTYPE_MAP[name]


def load_checkpoint(path: str, dtype: torch.dtype | None = None) -> CheckpointBundle:
    """从磁盘加载检查点并重建模型与 Adam 优化器。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点不存在: {path}")
    # weights_only=False：检查点内为我们自己写入的张量与基础容器
    blob = torch.load(path, map_location="cpu", weights_only=False)

    ckpt_dtype = _resolve_dtype(blob.get("dtype"))
    dtype = dtype or ckpt_dtype
    feature_names = list(blob["feature_names"])
    hidden_dim = int(blob["hidden_dim"])
    num_classes = int(blob["num_classes"])

    model = TwoLayerClassifier(
        input_dim=len(feature_names),
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    model.load_state_dict(blob["model_state"])
    model.to(dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(blob["optimizer_state"])
    _coerce_optimizer_dtype(optimizer, dtype)

    return CheckpointBundle(
        model=model,
        optimizer=optimizer,
        feature_names=feature_names,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dtype=dtype,
        source_path=os.path.abspath(path),
    )


def _coerce_optimizer_dtype(optimizer: torch.optim.Optimizer, dtype: torch.dtype) -> None:
    """加载后把 Adam 的 exp_avg / exp_avg_sq 转到目标 dtype。"""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(dtype)


def expand_bundle(
    bundle: CheckpointBundle,
    added_feature_names: list[str],
) -> CheckpointBundle:
    """扩列：模型追加零权重列，Adam 状态按参数对应关系迁移。

    - fc1.weight 追加零列，其 exp_avg / exp_avg_sq 同样追加零列；
    - 其余参数（fc1.bias、fc2.weight、fc2.bias）及其 Adam 状态原样保留；
    - step（含旧版 PyTorch 的 state[step] 张量）原样保留；
    - 优化器超参（lr、betas 等）原样保留。
    """
    added = list(added_feature_names)
    if any(name in bundle.feature_names for name in added):
        dup = [n for n in added if n in bundle.feature_names]
        raise ValueError(f"列名已存在，不能重复添加: {dup}")

    new_model = expand_model(bundle.model, added)
    n_add = len(added)

    old_optim = bundle.optimizer
    old_state = old_optim.state_dict()
    old_param_state = old_state["state"]

    # 新优化器，先拿到与新参数对象对应的干净 state 容器
    new_optimizer = torch.optim.Adam(
        new_model.parameters(),
        **{k: v for k, v in (old_optim.param_groups[0].items()) if k != "params"},
    )
    new_optim_state = new_optimizer.state_dict()
    new_param_state = new_optim_state["state"]

    old_named = dict(bundle.model.named_parameters())
    new_named = dict(new_model.named_parameters())
    old_param_ids = {id(p): name for name, p in old_named.items()}
    new_index = {name: idx for idx, (name, _) in enumerate(new_model.named_parameters())}
    old_in_features = bundle.model.fc1.in_features

    for param_index in range(len(list(new_model.parameters()))):
        new_param_state.setdefault(param_index, {})

    # state_dict()["state"] 以参数对象 id 为键（旧版 PyTorch 为顺序索引），
    # 统一通过参数名映射到新模型的参数索引，避免依赖 id/顺序假设。
    for key, state in old_param_state.items():
        param_name = old_param_ids.get(key)
        if param_name is None:
            # 旧版按顺序索引时，按旧模型参数顺序取名
            old_names = list(old_named.keys())
            param_name = old_names[key] if isinstance(key, int) and key < len(old_names) else None
        if param_name is None or param_name not in new_index:
            raise RuntimeError(f"无法迁移优化器状态，未知参数键: {key!r}")

        target_index = new_index[param_name]
        new_state: dict[str, Any] = {}
        for state_key, value in state.items():
            if (
                param_name == "fc1.weight"
                and torch.is_tensor(value)
                and value.dim() == 2
                and value.shape == (bundle.hidden_dim, old_in_features)
            ):
                # 与 fc1.weight 同形状的动量张量：追加零列
                zeros = value.new_zeros((value.shape[0], n_add))
                new_state[state_key] = torch.cat([value.clone(), zeros], dim=1)
            else:
                # 其余张量（含 0 维 step）与超参原样保留
                new_state[state_key] = value.clone() if torch.is_tensor(value) else value
        new_param_state[target_index] = new_state

    new_optimizer.load_state_dict(new_optim_state)

    return CheckpointBundle(
        model=new_model,
        optimizer=new_optimizer,
        feature_names=bundle.feature_names + added,
        hidden_dim=bundle.hidden_dim,
        num_classes=bundle.num_classes,
        dtype=bundle.dtype,
        source_path=bundle.source_path,
    )


def save_checkpoint_atomic(bundle: CheckpointBundle, path: str) -> None:
    """原子保存：先写临时文件再 rename，失败不影响原检查点。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(bundle.checkpoint_dict(), tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
