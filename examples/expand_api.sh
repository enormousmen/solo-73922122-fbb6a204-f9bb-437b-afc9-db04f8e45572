#!/usr/bin/env bash
# 扩列 API 示例：为运行中的模型追加一个振动测点列。
#
# 前提：docker compose up -d 已启动服务（默认 8000 端口）。
# 扩列只在内存中生效；确认无误后调用 /save 持久化，
# 再调用 /reload 或重启容器即可从新版本恢复。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== 1. 扩列前模型信息 =="
curl -s "${BASE_URL}/model" | python3 -m json.tool

echo
echo "== 2. 追加振动测点列 vibration =="
curl -s -X POST "${BASE_URL}/expand" \
    -H "Content-Type: application/json" \
    -d '{"new_feature_names": ["vibration"]}' | python3 -m json.tool

echo
echo "== 3. 用旧字段预测（vibration 缺失，自动补 0）=="
curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
            {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9}
          ]
        }' | python3 -m json.tool

echo
echo "== 4. 保存新版检查点（原子写，失败不影响旧文件）=="
curl -s -X POST "${BASE_URL}/save" \
    -H "Content-Type: application/json" \
    -d '{"path": "/app/checkpoints/model_v2.pt"}' | python3 -m json.tool

echo
echo "== 5. 从新版本重载（等价于重启后加载）=="
curl -s -X POST "${BASE_URL}/reload?path=/app/checkpoints/model_v2.pt" \
    | python3 -m json.tool
