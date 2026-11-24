#!/usr/bin/env bash
# 交付前关键自测：数值对照 + 全量 pytest。
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "== 1. 两处硬性数值对照（容差 1e-6）=="
"$PYTHON" - <<'EOF'
import torch
from app.checkpoint import load_checkpoint, expand_bundle
from app.data import align_rows
from app.service import ModelService

torch.set_num_threads(1)
OLD = ["temp", "pressure", "rpm", "current"]
g = torch.Generator().manual_seed(7)
rows = [dict(zip(OLD, torch.randn(4, generator=g).tolist())) for _ in range(12)]
labels = [i % 3 for i in range(12)]

base = load_checkpoint("checkpoints/model.pt")
exp = expand_bundle(load_checkpoint("checkpoints/model.pt"), ["vibration"])

def scores(b, rs):
    x, _ = align_rows(rs, b.feature_names, b.dtype)
    with torch.no_grad():
        return torch.softmax(b.model(x), dim=1)

e_score = (scores(base, rows) - scores(exp, rows)).abs().max().item()

plain = load_checkpoint("checkpoints/model.pt")
exp2 = expand_bundle(load_checkpoint("checkpoints/model.pt"), ["vibration"])
ModelService(plain).train_step(rows, labels)
ModelService(exp2).train_step(rows, labels)

max_e = e_score
for (n, p), (_, p2) in zip(plain.model.named_parameters(), exp2.model.named_parameters()):
    b = p2.detach()[:, :-1] if n == "fc1.weight" else p2.detach()
    max_e = max(max_e, (p.detach() - b).abs().max().item())

pp, pe = list(plain.model.parameters()), list(exp2.model.parameters())
idx = {n: i for i, (n, _) in enumerate(exp2.model.named_parameters())}
pidx = {n: i for i, (n, _) in enumerate(plain.model.named_parameters())}
for n in pidx:
    sp, se = plain.optimizer.state[pp[pidx[n]]], exp2.optimizer.state[pe[idx[n]]]
    for k in sp:
        a, b2 = sp[k], se[k]
        if n == "fc1.weight" and torch.is_tensor(a) and a.dim() == 2:
            b2 = b2[:, :-1]
        max_e = max(max_e, (a - b2).abs().max().item() if torch.is_tensor(a) else abs(a - b2))

print(f"分类分数误差: {e_score:.3e}；全部对照最大绝对误差: {max_e:.3e}（限值 1e-6）")
assert max_e <= 1e-6, "数值对照超限"
print("数值对照通过")
EOF

echo
echo "== 2. 全量自动化测试 =="
"$PYTHON" -m pytest tests/ -v
