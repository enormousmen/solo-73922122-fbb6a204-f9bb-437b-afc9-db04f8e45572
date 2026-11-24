"""API 端到端测试（TestClient，不占用真实端口）。"""
from __future__ import annotations

import os
import shutil

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


@pytest.fixture()
def client_on_copy(tmp_path, monkeypatch):
    # 检查点副本放在临时目录：便于原子替换与损坏原路径
    src = os.path.join("checkpoints", "model.pt")
    work = tmp_path / "work.pt"
    shutil.copy(src, work)
    monkeypatch.setenv("CHECKPOINT_PATH", str(work))
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    import importlib

    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, main, str(work)


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


# ---- 延迟物化与文件替换语义 ----------------------------------------


def test_predict_and_model_info_do_not_materialize_optimizer(client_on_copy):
    """只做预测/查模型信息：不读取动量张量；首次训练才物化。"""
    c, main, _ = client_on_copy
    assert main.service.bundle.optimizer_loaded is False

    assert c.get("/model").json()["optimizer_step"] == 200
    assert c.post("/predict", json={"instances": OLD_ROWS}).status_code == 200
    assert main.service.bundle.optimizer_loaded is False

    resp = c.post("/train", json={"instances": OLD_ROWS, "labels": [0, 1]})
    assert resp.status_code == 200
    assert main.service.bundle.optimizer_loaded is True


def test_atomic_replace_then_train_uses_load_time_snapshot(client_on_copy, tmp_path):
    """原路径被原子替换后，首次续训的动量仍来自加载时的那份文件。"""
    c, main, work = client_on_copy

    # 制备另一份检查点 B：多训 3 步（step=203），另存到临时路径
    from app.checkpoint import load_checkpoint, save_checkpoint_atomic
    from app.service import ModelService

    other = load_checkpoint(os.path.join("checkpoints", "model.pt"))
    svc_other = ModelService(other)
    for _ in range(3):
        svc_other.train_step(OLD_ROWS, [0, 1])
    b_path = tmp_path / "model_b.pt"
    save_checkpoint_atomic(other, str(b_path))

    # 替换前：step=200，记录预测结果
    assert c.get("/model").json()["optimizer_step"] == 200
    before = c.post("/predict", json={"instances": OLD_ROWS}).json()

    # 原子替换原路径上的检查点（模拟外部发布新模型）
    os.replace(b_path, work)

    # 内存模型不受替换影响：预测一致，步数仍是加载时的 200
    assert c.get("/model").json()["optimizer_step"] == 200
    assert c.post("/predict", json={"instances": OLD_ROWS}).json() == before

    # 首次续训：动量取自加载时钉住的文件 -> step 200+1=201，
    # 而不是替换后文件的 203+1
    resp = c.post("/train", json={"instances": OLD_ROWS, "labels": [0, 1]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["step"] == 201

    # 显式 /reload 后才切换到替换后的文件（step=203）
    resp = c.post("/reload")
    assert resp.status_code == 200, resp.text
    assert resp.json()["optimizer_step"] == 203
    resp = c.post("/train", json={"instances": OLD_ROWS, "labels": [0, 1]})
    assert resp.json()["step"] == 204


def test_failed_reload_keeps_predict_and_train(client_on_copy, tmp_path):
    """重载失败后，原模型仍可预测和续训。"""
    c, main, work = client_on_copy
    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"definitely not a checkpoint")

    resp = c.post("/reload", params={"path": str(bad)})
    assert resp.status_code == 500

    # 原模型仍可预测
    assert c.post("/predict", json={"instances": OLD_ROWS}).status_code == 200
    # 原模型仍可续训：step 200 -> 201
    resp = c.post("/train", json={"instances": OLD_ROWS, "labels": [0, 1]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["step"] == 201
    # 原路径仍可重载
    assert c.post("/reload").status_code == 200
    assert c.get("/model").json()["optimizer_step"] == 200
