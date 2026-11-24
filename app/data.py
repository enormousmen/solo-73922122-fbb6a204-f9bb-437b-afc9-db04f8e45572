"""样本数据按列名对齐。

输入样本为 {"列名": 数值} 形式的记录；对齐到模型当前的
feature_names：缺失列补 0，多余列忽略，顺序由模型决定。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def align_rows(
    rows: Iterable[Mapping[str, Any]],
    feature_names: list[str],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[list[str]]]:
    """把记录列表对齐为 (N, len(feature_names)) 的张量。

    返回 (张量, 每行被补 0 的列名列表)。
    """
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return torch.empty((0, len(feature_names)), dtype=dtype), []

    out = torch.zeros((n, len(feature_names)), dtype=dtype)
    filled: list[list[str]] = [[] for _ in range(n)]
    index = {name: j for j, name in enumerate(feature_names)}

    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"第 {i} 条样本必须是对象/字典，实际为 {type(row)!r}")
        for key, value in row.items():
            j = index.get(key)
            if j is None:
                continue  # 模型不认识的列直接忽略
            out[i, j] = float(value)
        missing = [name for name in feature_names if name not in row]
        filled[i] = missing
    return out, filled


def align_labels(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    labels = list(labels)
    tensor = torch.tensor(labels, dtype=torch.long)
    if tensor.numel() > 0 and (int(tensor.min()) < 0 or int(tensor.max()) >= num_classes):
        raise ValueError(
            f"标签必须落在 [0, {num_classes - 1}]，实际区间为 "
            f"[{int(tensor.min())}, {int(tensor.max())}]"
        )
    return tensor
