"""两层分类模型定义与列扩展逻辑。

模型结构固定为两层全连接：
    hidden = ReLU(X @ W1.T + b1)
    logits = hidden @ W2.T + b2

扩列（新增振动测点）时只在 W1 末尾追加零初始化的列，
W2、所有偏置及参数顺序保持不变，因此旧样本（新列补 0）
在前向上与扩列前逐位相等。
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class TwoLayerClassifier(nn.Module):
    """两层全连接分类器（一个隐藏层 + 一个输出层）。"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

    def clone(self) -> "TwoLayerClassifier":
        return copy.deepcopy(self)


def expand_model(
    model: TwoLayerClassifier,
    added_feature_names: list[str],
) -> TwoLayerClassifier:
    """在输入维度尾部追加新列，新列对应权重以 0 初始化。

    返回新模型；fc2、偏置及参数顺序完全保留，
    fc1 权重为 [旧权重 | 零列]，偏置不变。
    """
    if not added_feature_names:
        raise ValueError("added_feature_names 不能为空")

    old_fc1 = model.fc1
    in_dim = old_fc1.in_features
    n_add = len(added_feature_names)
    new_model = TwoLayerClassifier(
        input_dim=in_dim + n_add,
        hidden_dim=model.hidden_dim,
        num_classes=model.num_classes,
    )

    with torch.no_grad():
        # 原有权重逐位保留，新增输入列对应的权重为 0
        new_model.fc1.weight.copy_(
            torch.cat(
                [old_fc1.weight.detach().clone(),
                 old_fc1.weight.new_zeros((old_fc1.out_features, n_add))],
                dim=1,
            )
        )
        new_model.fc1.bias.copy_(old_fc1.bias.detach().clone())
        new_model.fc2.weight.copy_(model.fc2.weight.detach().clone())
        new_model.fc2.bias.copy_(model.fc2.bias.detach().clone())
    return new_model
