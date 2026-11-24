"""API 端到端测试（TestClient，不占用真实端口）。"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 每个测试使用独立的检查点副本，避免互相污染
    src = os.path.join("checkpoints", "model.pt")
    monkeypatch.setenv("CHECKPOINT_PATH", src)
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    # 延迟导入，确保环境变量先生效
    import importlib

    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, main


OLD_ROWS = [
    {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
    {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9},
]


def test_health_and_model_info(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}
    info = c.get("/model").json()
    assert info["feature_names"] == ["temp", "pressure", "rpm", "current"]
    assert info["input_dim"] == 4
    assert info["optimizer_step"] == 200


def test_predict_old_fields_and_zero_fill(client):
    c, _ = client
    resp = c.post("/predict", json={"instances": OLD_ROWS})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2
    item = body["predictions"][0]
    assert set(item.keys()) == {"label", "score", "probabilities", "zero_filled_columns"}
    assert len(item["probabilities"]) == 3
    assert abs(sum(item["probabilities"]) - 1.0) < 1e-6
    assert item["zero_filled_columns"] == []  # 扩列前没有缺失列


def test_expand_predict_train_save_reload_flow(client, tmp_path):
    c, main = client

    # 1) 扩列
    resp = c.post("/expand", json={"new_feature_names": ["vibration"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["feature_names"][-1] == "vibration"
    assert resp.json()["input_dim"] == 5

    # 2) 旧字段预测：vibration 自动补 0
    resp_old = c.post("/predict", json={"instances": OLD_ROWS}).json()
    assert resp_old["feature_names"] == [
        "temp", "pressure", "rpm", "current", "vibration"
    ]
    assert resp_old["predictions"][0]["zero_filled_columns"] == ["vibration"]

    # 3) 带新测点的样本续训
    rows_vib = [
        {**row, "vibration": 0.25 * (i + 1)} for i, row in enumerate(OLD_ROWS * 2)
    ]
    labels = [0, 1, 2, 1]
    resp = c.post("/train", json={"instances": rows_vib, "labels": labels})
    assert resp.status_code == 200, resp.text
    train_body = resp.json()
    assert train_body["step"] == 201
    assert train_body["loss_after"] <= train_body["loss_before"] + 1e-9

    # 4) 保存新版本到独立路径
    v2 = str(tmp_path / "model_v2.pt")
    resp = c.post("/save", json={"path": v2})
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert os.path.exists(v2)

    # 5) 模拟重启：从新检查点重新加载
    resp = c.post("/reload", params={"path": v2})
    assert resp.status_code == 200, resp.text
    rj = resp.json()
    assert rj["feature_names"][-1] == "vibration"
    assert rj["optimizer_step"] == 201

    # 6) 重启后旧字段仍可预测
    resp = c.post("/predict", json={"instances": OLD_ROWS})
    assert resp.status_code == 200
    assert resp.json()["predictions"][0]["zero_filled_columns"] == ["vibration"]

    # 7) 重启后可按新列继续训练
    resp = c.post("/train", json={"instances": rows_vib, "labels": labels})
    assert resp.status_code == 200 and resp.json()["step"] == 202


def test_save_failure_does_not_break_service(client, tmp_path):
    c, _ = client
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    bad = str(blocker / "nested" / "x.pt")
    resp = c.post("/save", json={"path": bad})
    assert resp.status_code == 500
    # 服务仍可预测，原检查点仍可重载
    assert c.post("/predict", json={"instances": OLD_ROWS}).status_code == 200
    assert c.post("/reload").status_code == 200
    assert c.get("/model").json()["input_dim"] == 4


def test_reject_duplicate_and_empty_expand(client):
    c, _ = client
    assert c.post("/expand", json={"new_feature_names": ["temp"]}).status_code == 422
    assert c.post("/expand", json={"new_feature_names": []}).status_code == 422


def test_label_validation(client):
    c, _ = client
    resp = c.post(
        "/train",
        json={"instances": OLD_ROWS, "labels": [0, 9]},
    )
    assert resp.status_code == 422


def test_train_count_mismatch(client):
    c, _ = client
    resp = c.post(
        "/train",
        json={"instances": OLD_ROWS, "labels": [0]},
    )
    assert resp.status_code == 422
