#!/usr/bin/env bash
# 替换文件后续训示例。
#
# 场景：服务运行期间，原路径上的检查点被另一进程“原子替换”
# （先写好新文件再 mv 覆盖，与 /save 的原子写同一机制）。
# 服务加载权重时已用 mmap 钉住当时的那份文件，因此替换后首次
# 续训取用的动量仍来自它（step 接续加载时的 200），而不是替换
# 后的新文件；显式 /reload 之后才切换到新文件。
#
# 前提：docker compose up -d 已启动服务，./checkpoints 挂载进容器。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
# CKPT_DIR：宿主机上的检查点目录（mv 在这里执行）
# CKPT_API_DIR：服务进程看到的检查点目录（容器内为 /app/checkpoints）
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
CKPT_API_DIR="${CKPT_API_DIR:-/app/checkpoints}"
MAIN_CKPT="${CKPT_DIR}/model.pt"
NEW_CKPT="${CKPT_DIR}/model_v2.pt"

echo "== 1. 当前模型（启动时加载的检查点，step=200）=="
curl -s "${BASE_URL}/model" | python3 -m json.tool

echo
echo "== 2. 制备一份更新的检查点：续训 3 步后另存为 model_v2.pt =="
for i in 1 2 3; do
  curl -s -X POST "${BASE_URL}/train" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
            {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9}
          ],
          "labels": [0, 1]
        }' | python3 -m json.tool
done
curl -s -X POST "${BASE_URL}/save" \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"${CKPT_API_DIR}/model_v2.pt\"}" | python3 -m json.tool

echo
echo "== 3. 回退到原检查点（step=200 的延迟物化快照）=="
curl -s -X POST "${BASE_URL}/reload?path=${CKPT_API_DIR}/model.pt" | python3 -m json.tool

echo
echo "== 4. 在宿主机上原子替换原路径文件（model_v2.pt -> model.pt）=="
mv -v "${NEW_CKPT}" "${MAIN_CKPT}"

echo
echo "== 5. 替换后首次续训：step 从加载时的 200 变为 201 =="
echo "   （动量取自加载权重时钉住的那份文件，而非替换后的 step=203 文件）"
curl -s -X POST "${BASE_URL}/train" \
  -H "Content-Type: application/json" \
  -d '{
        "instances": [
          {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
          {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9}
        ],
        "labels": [0, 1]
      }' | python3 -m json.tool

echo
echo "== 6. 显式 /reload 后才切换到替换后的文件（step=203，续训得 204）=="
curl -s -X POST "${BASE_URL}/reload" | python3 -m json.tool
curl -s -X POST "${BASE_URL}/train" \
  -H "Content-Type: application/json" \
  -d '{
        "instances": [
          {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
          {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9}
        ],
        "labels": [0, 1]
      }' | python3 -m json.tool
