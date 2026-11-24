"""FastAPI 请求/响应模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # 每条样本为 {列名: 数值}；模型需要但未提供的列自动补 0
    instances: list[dict[str, float]] = Field(min_length=1)


class PredictedItem(BaseModel):
    label: int
    score: float
    probabilities: list[float]
    zero_filled_columns: list[str]


class PredictResponse(BaseModel):
    feature_names: list[str]
    predictions: list[PredictedItem]


class TrainRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


class TrainResponse(BaseModel):
    step: int
    loss_before: float
    loss_after: float
    num_samples: int
    feature_names: list[str]


class ExpandRequest(BaseModel):
    new_feature_names: list[str] = Field(min_length=1)


class ExpandResponse(BaseModel):
    feature_names: list[str]
    input_dim: int
    added: list[str]


class ModelInfoResponse(BaseModel):
    feature_names: list[str]
    input_dim: int
    hidden_dim: int
    num_classes: int
    dtype: str
    optimizer_step: int
    source_path: str


class SaveRequest(BaseModel):
    path: str | None = None  # 为空则覆盖当前检查点路径


class SaveResponse(BaseModel):
    path: str
    ok: bool


class ReloadResponse(BaseModel):
    path: str
    feature_names: list[str]
    input_dim: int
    optimizer_step: int


class ErrorMessage(BaseModel):
    detail: Any
