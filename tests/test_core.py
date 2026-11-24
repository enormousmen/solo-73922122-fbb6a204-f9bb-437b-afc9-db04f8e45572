"""关键自测。

覆盖需求中的硬性条款：
1. 扩列后续训前，旧样本分类分数与扩列前一致（abs err <= 1e-6）；
2. 同一批旧样本各训练一步，扩列/未扩列模型的原参数与对应
   Adam 状态一致（abs err <= 1e-6）；
3. 新列零权重、新列读数可更新新增权重；
4. 缺新列补 0、按列名对齐、多余列忽略；
5. 保存失败不影响原检查点，成功后重载可按旧字段预测、按新列续训；
6. 检查点往返后 Adam step 等状态保留；
7. 延迟物化（defer_optimizer）：启动只读权重/列名/步数，动量首次
   使用时才读入；原子替换后动量仍来自加载时的那份文件；重载失败
   后原模型仍可预测和续训；与原加载方式逐位一致。
"""
from __future__ import annotations

import copy
import os
import shutil

import pytest
import torch

from app.checkpoint import expand_bundle, load_checkpoint, save_checkpoint_atomic
from app.data import align_rows
from app.model import expand_model
from app.service import ModelService

torch.set_num_threads(1)
TOL = 1e-6

OLD_FEATURES = ["temp", "pressure", "rpm", "current"]
NEW_COL = "vibration"


@pytest.fixture(scope="module")
def bundle():
    return load_checkpoint(os.path.join("checkpoints", "model.pt"))


@pytest.fixture(scope="module")
def old_rows():
    g = torch.Generator().manual_seed(7)
    return [
        dict(zip(OLD_FEATURES, (torch.randn(len(OLD_FEATURES), generator=g)).tolist()))
        for _ in range(12)
    ]


def _scores(model, rows, feature_names, dtype):
    x, _ = align_rows(rows, feature_names, dtype)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)


def test_expand_keeps_old_scores_identical(bundle, old_rows):
    """条款 1：扩列后续训前，旧样本分类分数保持一致。"""
    expanded = expand_bundle(bundle, [NEW_COL])

    before = _scores(bundle.model, old_rows, bundle.feature_names, bundle.dtype)
    # 扩列后的旧样本不带 vibration 列 -> 自动补 0
    after = _scores(expanded.model, old_rows, expanded.feature_names, expanded.dtype)

    assert before.shape == after.shape
    err = (before - after).abs().max().item()
    assert err <= TOL, f"分类分数最大绝对误差 {err} 超过 {TOL}"


def test_expand_zero_initializes_new_column(bundle):
    expanded = expand_bundle(bundle, [NEW_COL])
    w = expanded.model.fc1.weight.detach()
    assert w.shape == (bundle.hidden_dim, len(OLD_FEATURES) + 1)
    assert torch.equal(w[:, -1], torch.zeros_like(w[:, -1]))
    # 原有权重逐位相同
    assert torch.equal(w[:, :-1], bundle.model.fc1.weight.detach())
    # 偏置与第二层完全不变
    assert torch.equal(expanded.model.fc1.bias.detach(), bundle.model.fc1.bias.detach())
    assert torch.equal(expanded.model.fc2.weight.detach(), bundle.model.fc2.weight.detach())
    assert torch.equal(expanded.model.fc2.bias.detach(), bundle.model.fc2.bias.detach())


def test_train_step_matches_between_expanded_and_plain(bundle, old_rows):
    """条款 2：同一批旧样本各训一步，原参数与 Adam 状态一致。"""
    labels = [i % bundle.num_classes for i in range(len(old_rows))]

    plain = load_checkpoint(os.path.join("checkpoints", "model.pt"))
    expanded = expand_bundle(load_checkpoint(os.path.join("checkpoints", "model.pt")), [NEW_COL])

    out_plain = ModelService(plain).train_step(old_rows, labels)
    # 扩列模型喂同样的旧字段，vibration 缺省补 0
    out_exp = ModelService(expanded).train_step(old_rows, labels)

    assert abs(out_plain["loss_before"] - out_exp["loss_before"]) <= TOL
    assert abs(out_plain["loss_after"] - out_exp["loss_after"]) <= TOL

    # 原参数一致
    for (n_p, p_p), (n_e, p_e) in zip(
        plain.model.named_parameters(), expanded.model.named_parameters()
    ):
        assert n_p == n_e or n_e == n_p
        if n_e == "fc1.weight":
            err = (p_p.detach() - p_e.detach()[:, :-1]).abs().max().item()
            assert err <= TOL, f"fc1.weight 原列误差 {err}"
        else:
            err = (p_p.detach() - p_e.detach()).abs().max().item()
            assert err <= TOL, f"{n_e} 参数误差 {err}"

    # Adam 状态一致（exp_avg / exp_avg_sq / step）
    plain_state = plain.optimizer.state
    exp_state = expanded.optimizer.state
    params_plain = list(plain.model.parameters())
    params_exp = list(expanded.model.parameters())
    idx_exp = {name: i for i, (name, _) in enumerate(expanded.model.named_parameters())}
    idx_plain = {name: i for i, (name, _) in enumerate(plain.model.named_parameters())}

    for name in idx_plain:
        s_p = plain_state[params_plain[idx_plain[name]]]
        s_e = exp_state[params_exp[idx_exp[name]]]
        assert set(s_p.keys()) == set(s_e.keys())
        for key in s_p:
            v_p, v_e = s_p[key], s_e[key]
            if name == "fc1.weight" and torch.is_tensor(v_p) and v_p.dim() == 2:
                err = (v_p - v_e[:, :-1]).abs().max().item()
            elif torch.is_tensor(v_p):
                err = (v_p - v_e).abs().max().item()
            else:
                err = abs(v_p - v_e)
            assert err <= TOL, f"Adam 状态 {name}.{key} 误差 {err}"

    # 新列在纯旧样本训练后仍为 0（梯度也是 0）
    assert torch.equal(
        expanded.model.fc1.weight.detach()[:, -1],
        torch.zeros(bundle.hidden_dim, dtype=bundle.dtype),
    )


def test_new_readings_update_new_weights(bundle, old_rows):
    """带新测点读数的样本能更新新增权重。"""
    expanded = expand_bundle(load_checkpoint(os.path.join("checkpoints", "model.pt")), [NEW_COL])
    labels = [i % bundle.num_classes for i in range(len(old_rows))]
    rows_with_vib = [{**row, NEW_COL: 0.37 * (i + 1)} for i, row in enumerate(old_rows)]

    before = expanded.model.fc1.weight.detach()[:, -1].clone()
    ModelService(expanded).train_step(rows_with_vib, labels)
    after = expanded.model.fc1.weight.detach()[:, -1]

    assert torch.equal(before, torch.zeros_like(before))
    assert not torch.equal(after, before), "新测点权重未被更新"
    assert after.abs().max().item() > 0.0


def test_align_by_name_missing_zero_and_extra_ignored(bundle):
    rows = [
        {"rpm": 1.0, "temp": 2.0, "unknown_col": 9.9},  # 缺 pressure/current，多未知列
        {name: float(i) for i, name in enumerate(OLD_FEATURES)},
    ]
    x, missing = align_rows(rows, OLD_FEATURES, bundle.dtype)
    assert x.shape == (2, 4)
    # 第一行按列名落到正确位置
    assert x[0, OLD_FEATURES.index("temp")].item() == 2.0
    assert x[0, OLD_FEATURES.index("rpm")].item() == 1.0
    assert x[0, OLD_FEATURES.index("pressure")].item() == 0.0
    assert x[0, OLD_FEATURES.index("current")].item() == 0.0
    assert set(missing[0]) == {"pressure", "current"}
    assert missing[1] == []

    # 扩列后旧样本缺新列补 0
    expanded = expand_bundle(bundle, [NEW_COL])
    x2, missing2 = align_rows(rows[:1], expanded.feature_names, expanded.dtype)
    assert x2.shape == (1, 5)
    assert x2[0, -1].item() == 0.0
    assert NEW_COL in missing2[0]


def test_save_failure_keeps_original_checkpoint(bundle, tmp_path):
    """保存失败仍可用原检查点。"""
    src = os.path.join("checkpoints", "model.pt")
    original_blob = torch.load(src, map_location="cpu", weights_only=False)

    expanded = expand_bundle(load_checkpoint(src), [NEW_COL])
    bad_path = str(tmp_path / "no_such_dir" / "nested" / "model_v2.pt")
    # 让父目录创建也失败：把中间路径占成文件
    blocker = tmp_path / "no_such_dir"
    blocker.write_text("x")
    with pytest.raises(OSError):
        save_checkpoint_atomic(expanded, bad_path)

    # 原检查点完好且可加载
    reloaded = load_checkpoint(src)
    assert reloaded.feature_names == OLD_FEATURES
    for (k, v), (k2, v2) in zip(
        original_blob["model_state"].items(), reloaded.model.state_dict().items()
    ):
        assert k == k2 and torch.equal(v, v2)


def test_save_and_reload_roundtrip_supports_old_predict_and_retrain(bundle, old_rows, tmp_path):
    """成功保存后重启等价场景：重载 -> 旧字段预测 -> 按新列续训。"""
    src = os.path.join("checkpoints", "model.pt")
    expanded = expand_bundle(load_checkpoint(src), [NEW_COL])
    labels = [i % bundle.num_classes for i in range(len(old_rows))]
    rows_with_vib = [{**row, NEW_COL: 0.1 * (i - 3)} for i, row in enumerate(old_rows)]
    ModelService(expanded).train_step(rows_with_vib, labels)

    v2_path = tmp_path / "model_v2.pt"
    save_checkpoint_atomic(expanded, str(v2_path))

    # 模拟重启：从磁盘重新加载
    reloaded = load_checkpoint(str(v2_path))
    assert reloaded.feature_names == OLD_FEATURES + [NEW_COL]

    # 旧字段照常预测（vibration 补 0），与保存前内存模型结果一致
    s_before = _scores(expanded.model, old_rows, expanded.feature_names, expanded.dtype)
    s_after = _scores(reloaded.model, old_rows, reloaded.feature_names, reloaded.dtype)
    assert (s_before - s_after).abs().max().item() <= TOL

    # Adam step 保留，且可继续训练
    step_before = int(reloaded.optimizer.state[list(reloaded.model.parameters())[0]]["step"])
    assert step_before == 201  # 初始 200 步 + 保存前续训 1 步
    result = ModelService(reloaded).train_step(rows_with_vib, labels)
    step_after = int(reloaded.optimizer.state[list(reloaded.model.parameters())[0]]["step"])
    assert step_after == step_before + 1
    assert result["step"] == step_after


def test_repeated_expand_multiple_columns(bundle):
    b2 = expand_bundle(bundle, ["vibration_a", "vibration_b"])
    assert b2.feature_names == OLD_FEATURES + ["vibration_a", "vibration_b"]
    w = b2.model.fc1.weight.detach()
    assert torch.equal(w[:, -2:], torch.zeros_like(w[:, -2:]))
    with pytest.raises(ValueError):
        expand_bundle(b2, ["vibration_a"])  # 重名拒绝


def test_duplicate_column_rejected(bundle):
    with pytest.raises(ValueError):
        expand_bundle(bundle, [OLD_FEATURES[0]])


# ---- 延迟物化（defer_optimizer） ----------------------------------


def _assert_same_params_and_adam(b1, b2, tol=TOL):
    """两个同形状 bundle 的权重与 Adam 状态逐位对照。"""
    for (n1, p1), (n2, p2) in zip(
        b1.model.named_parameters(), b2.model.named_parameters()
    ):
        assert n1 == n2
        err = (p1.detach() - p2.detach()).abs().max().item()
        assert err <= tol, f"参数 {n1} 最大绝对误差 {err}"
    params1 = list(b1.model.parameters())
    params2 = list(b2.model.parameters())
    state1, state2 = b1.optimizer.state, b2.optimizer.state
    for p1, p2 in zip(params1, params2):
        s1, s2 = state1[p1], state2[p2]
        assert set(s1.keys()) == set(s2.keys())
        for key in s1:
            v1, v2 = s1[key], s2[key]
            if torch.is_tensor(v1):
                err = (v1 - v2).abs().max().item()
                assert err <= tol, f"Adam 状态 {n1}.{key} 最大绝对误差 {err}"
            else:
                assert v1 == v2


def test_lazy_load_defers_optimizer_until_train(old_rows):
    """延迟物化：启动只读权重/列名/步数，首次训练才读动量。"""
    src = os.path.join("checkpoints", "model.pt")
    b = load_checkpoint(src, defer_optimizer=True)
    assert not b.optimizer_loaded
    assert b.optimizer_step == 200  # 步数加载时即可读，不触发物化

    svc = ModelService(b)
    preds = svc.predict(old_rows)  # 预测不需要动量
    assert len(preds) == len(old_rows)
    assert not b.optimizer_loaded

    labels = [i % b.num_classes for i in range(len(old_rows))]
    out = svc.train_step(old_rows, labels)  # 首次训练才物化
    assert b.optimizer_loaded
    assert out["step"] == 201
    assert b.optimizer_step == 201


def test_lazy_first_train_matches_eager(old_rows):
    """同一运行精度与输入：懒加载首次续训与原加载方式逐位一致。"""
    src = os.path.join("checkpoints", "model.pt")
    labels = [i % 3 for i in range(len(old_rows))]

    eager = load_checkpoint(src)
    lazy = load_checkpoint(src, defer_optimizer=True)
    ModelService(eager).train_step(old_rows, labels)
    ModelService(lazy).train_step(old_rows, labels)

    _assert_same_params_and_adam(eager, lazy)


def test_lazy_expand_matches_eager(old_rows):
    """懒加载后首次扩列 + 续训，与原加载方式逐位一致。"""
    src = os.path.join("checkpoints", "model.pt")
    labels = [i % 3 for i in range(len(old_rows))]

    eager = expand_bundle(load_checkpoint(src), [NEW_COL])
    lazy = expand_bundle(load_checkpoint(src, defer_optimizer=True), [NEW_COL])
    assert lazy.optimizer_loaded  # 扩列触发物化
    _assert_same_params_and_adam(eager, lazy)

    ModelService(eager).train_step(old_rows, labels)
    ModelService(lazy).train_step(old_rows, labels)
    _assert_same_params_and_adam(eager, lazy)


def test_lazy_momentum_from_load_time_file_after_atomic_replace(tmp_path, old_rows):
    """原路径被原子替换后，首次取用的动量仍来自加载时的那份文件。"""
    src = os.path.join("checkpoints", "model.pt")
    work = tmp_path / "work.pt"
    shutil.copy(src, work)
    lazy = load_checkpoint(str(work), defer_optimizer=True)
    labels = [i % lazy.num_classes for i in range(len(old_rows))]

    # 另一份检查点 B：多训 3 步（step=203，动量不同）
    other = load_checkpoint(src)
    svc_other = ModelService(other)
    for _ in range(3):
        svc_other.train_step(old_rows, labels)
    assert other.optimizer_step == 203
    b_path = tmp_path / "model_b.pt"
    save_checkpoint_atomic(other, str(b_path))

    os.replace(b_path, work)  # 原子替换原路径

    # 首次物化：动量来自加载权重时的文件（step=200），不是替换后的 B
    assert lazy.optimizer_step == 200
    ModelService(lazy).train_step(old_rows, labels)
    assert lazy.optimizer_step == 201

    # 与“原方式加载旧文件 + 训一步”逐位一致
    ref = load_checkpoint(src)
    ModelService(ref).train_step(old_rows, labels)
    _assert_same_params_and_adam(ref, lazy)


def test_lazy_dtype_conversion_deferred(old_rows):
    """运行精度与文件精度不同：转换随物化一起推迟，结果与原方式一致。"""
    src = os.path.join("checkpoints", "model.pt")  # 文件为 float32
    labels = [i % 3 for i in range(len(old_rows))]

    eager = load_checkpoint(src, dtype=torch.float64)
    lazy = load_checkpoint(src, dtype=torch.float64, defer_optimizer=True)
    assert lazy.model.fc1.weight.dtype == torch.float64  # 权重已转运行精度
    assert not lazy.optimizer_loaded  # 动量尚未读取，也未转换

    ModelService(eager).train_step(old_rows, labels)
    ModelService(lazy).train_step(old_rows, labels)
    _assert_same_params_and_adam(eager, lazy)
    for state in lazy.optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.dtype == torch.float64


def test_lazy_save_writes_complete_checkpoint(tmp_path):
    """懒加载后 /save 触发物化，写出的检查点可按原格式完整加载。"""
    src = os.path.join("checkpoints", "model.pt")
    lazy = load_checkpoint(src, defer_optimizer=True)
    assert not lazy.optimizer_loaded

    out = tmp_path / "saved.pt"
    save_checkpoint_atomic(lazy, str(out))
    assert lazy.optimizer_loaded  # 保存触发物化

    reloaded = load_checkpoint(str(out))  # 原方式加载
    assert reloaded.feature_names == OLD_FEATURES
    assert reloaded.optimizer_step == 200
    _assert_same_params_and_adam(load_checkpoint(src), reloaded)


def test_failed_reload_keeps_old_bundle_usable(tmp_path, old_rows):
    """重载失败后，原模型仍可预测和续训（动量取自加载时的快照）。"""
    src = os.path.join("checkpoints", "model.pt")
    work = tmp_path / "work.pt"
    shutil.copy(src, work)
    svc = ModelService(load_checkpoint(str(work), defer_optimizer=True))
    labels = [i % 3 for i in range(len(old_rows))]

    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"definitely not a checkpoint")
    with pytest.raises(Exception):
        svc.reload(str(bad))

    # 原模型仍可预测
    assert len(svc.predict(old_rows)) == len(old_rows)

    # 即便原路径随后也被破坏，续训仍从加载时钉住的文件快照取动量
    os.replace(bad, work)
    out = svc.train_step(old_rows, labels)
    assert out["step"] == 201

    # 与“原方式加载 + 训一步”逐位一致
    ref = load_checkpoint(src)
    ModelService(ref).train_step(old_rows, labels)
    _assert_same_params_and_adam(ref, svc.bundle)
