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

延迟物化（defer_optimizer）：
内存紧张的巡检节点上，启动 / 重载只读模型权重、列名与 Adam 步数
（供 /predict 与 /model 使用）；动量张量（exp_avg / exp_avg_sq）
以 mmap 形式留在原检查点文件里，首次 /train、/expand 或 /save
时才读入内存并做精度转换（运行精度与文件精度不同时也一样）。
mmap 钉住了加载时的文件 inode：原路径即使被原子替换，首次取用
的动量仍来自加载当前权重时用的那份文件。
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from typing import Any

import torch

from .model import TwoLayerClassifier, expand_model

CHECKPOINT_VERSION = 1


class CheckpointBundle:
    """内存中活动的模型 + （可延迟物化的）Adam 优化器 + 元数据。

    以 defer_optimizer 方式加载时 optimizer 尚未物化：动量张量以
    mmap 形式留在检查点文件中，首次访问 optimizer（训练 / 扩列 /
    保存）时才读入并转换精度；/predict 与 /model 不触发物化。
    """

    def __init__(
        self,
        *,
        model: TwoLayerClassifier,
        feature_names: list[str],
        hidden_dim: int,
        num_classes: int,
        dtype: torch.dtype,
        source_path: str,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_step: int = 0,
        pending_optimizer_state: dict[str, Any] | None = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dtype = dtype
        self.source_path = source_path
        self._optimizer = optimizer
        self._pending_optimizer_state = pending_optimizer_state
        self._deferred_step = optimizer_step

    # ---- 优化器（延迟物化） --------------------------------------
    @property
    def optimizer_loaded(self) -> bool:
        """动量是否已物化到内存。"""
        return self._optimizer is not None

    @property
    def optimizer_step(self) -> int:
        """Adam 步数：未物化时返回加载时读到的值，不触发物化。"""
        if self._optimizer is not None:
            return _read_optimizer_step(self._optimizer)
        return self._deferred_step

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.ensure_optimizer()

    def ensure_optimizer(self) -> torch.optim.Optimizer:
        """首次需要时物化 Adam：读入动量张量并转到运行精度。

        物化完成后释放对检查点文件的 mmap 引用。
        """
        if self._optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())
            if self._pending_optimizer_state:
                optimizer.load_state_dict(self._pending_optimizer_state)
            _coerce_optimizer_dtype(optimizer, self.dtype)
            self._optimizer = optimizer
            self._pending_optimizer_state = None
        return self._optimizer

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


def load_checkpoint(
    path: str,
    dtype: torch.dtype | None = None,
    defer_optimizer: bool = False,
) -> CheckpointBundle:
    """从磁盘加载检查点并重建模型（可选延迟物化 Adam）。

    defer_optimizer=False（原方式）：一次性读入权重与全部 Adam 状态。
    defer_optimizer=True：只读权重、列名与 Adam 步数；动量张量以
    mmap 留在文件中，首次训练 / 扩列 / 保存时才读入并转换精度。
    两种方式在相同运行精度与输入下，首次续训 / 扩列后的权重与
    Adam 状态逐位一致。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点不存在: {path}")
    # weights_only=False：检查点内为我们自己写入的张量与基础容器
    if defer_optimizer:
        blob = _load_maybe_mmap(path)
    else:
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

    if defer_optimizer:
        # 只保留优化器状态（mmap 支撑，未读入常驻内存），其余引用释放
        optimizer_state = blob["optimizer_state"]
        return CheckpointBundle(
            model=model,
            feature_names=feature_names,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dtype=dtype,
            source_path=os.path.abspath(path),
            optimizer=None,
            optimizer_step=_read_step_from_state_dict(optimizer_state),
            pending_optimizer_state=optimizer_state,
        )

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(blob["optimizer_state"])
    _coerce_optimizer_dtype(optimizer, dtype)

    return CheckpointBundle(
        model=model,
        feature_names=feature_names,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dtype=dtype,
        source_path=os.path.abspath(path),
        optimizer=optimizer,
    )


def _load_maybe_mmap(path: str) -> dict[str, Any]:
    """以 mmap 方式打开检查点：存储页按需读入，不占常驻内存。

    mmap 钉住文件 inode：路径被原子替换后，从该句柄读到的仍是
    打开时的那份文件。老版本 torch（无 mmap 参数）或旧格式文件
    退回全量读取：数值与“同一份文件”语义不变，只失去省内存效果。
    """
    try:
        if zipfile.is_zipfile(path):
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # 老版本 torch 没有 mmap 参数
        pass
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_step_from_state_dict(optimizer_state: dict[str, Any]) -> int:
    """从 optimizer.state_dict() 结构中读取 Adam 步数（不物化优化器）。"""
    state = optimizer_state.get("state") or {}
    if not state:
        return 0
    step = next(iter(state.values())).get("step", 0)
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)


def _read_optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    """兼容不同 PyTorch 版本读取 Adam 的 step。"""
    state = optimizer.state
    if not state:
        return 0
    step = next(iter(state.values())).get("step", 0)
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)


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

    若 bundle 的优化器尚未物化，此处首次触发物化（读取动量）。
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
    """原子保存：先写临时文件再 rename，失败不影响原检查点。

    若 bundle 的优化器尚未物化，此处首次触发物化（读取动量），
    写出的仍是含完整 Adam 状态的原格式检查点。
    """
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
