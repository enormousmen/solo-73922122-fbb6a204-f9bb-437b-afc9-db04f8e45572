#!/usr/bin/env bash
# 只做预测的示例：内存紧张的巡检节点启动后，/model 与 /predict
# 只使用权重、列名与 Adam 步数；动量张量以 mmap 留在检查点文件
# 中，首次 /train、/expand 或 /save 时才读入内存。
#
# 前提：docker compose up -d 已启动服务（默认 8000 端口）。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== 1. 模型信息：列名、维度、dtype、Adam 步数（不触发动量读取）=="
curl -s "${BASE_URL}/model" | python3 -m json.tool

echo
echo "== 2. 只做预测：未改造设备的原四字段样本 =="
curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
            {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9}
          ]
        }' | python3 -m json.tool

echo
echo "== 3. 只做预测：缺列自动补 0、多余列忽略 =="
curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"rpm": 1.0, "temp": 2.0, "unknown_col": 9.9}
          ]
        }' | python3 -m json.tool
