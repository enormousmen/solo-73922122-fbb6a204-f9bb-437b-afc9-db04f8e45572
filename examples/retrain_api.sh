#!/usr/bin/env bash
# 续训 API 示例：用含新测点 vibration 读数的样本再训练一步。
#
# 前提：已执行 expand_api.sh（模型当前输入为
# temp, pressure, rpm, current, vibration）。
# 未改造设备的样本继续只传原字段，缺的 vibration 列补 0。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== 1. 未改造设备样本：仅原字段，vibration 自动补 0 =="
curl -s -X POST "${BASE_URL}/train" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
            {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9},
            {"temp": 0.2, "pressure": 0.0, "rpm": 0.7, "current": -0.5},
            {"temp": -0.6, "pressure": 1.1, "rpm": -0.9, "current": 0.3}
          ],
          "labels": [0, 1, 2, 1]
        }' | python3 -m json.tool

echo
echo "== 2. 改造后设备样本：带 vibration 读数，更新新增权重 =="
curl -s -X POST "${BASE_URL}/train" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1, "vibration": 0.25},
            {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9, "vibration": 0.51},
            {"temp": 0.2, "pressure": 0.0, "rpm": 0.7, "current": -0.5, "vibration": -0.37},
            {"temp": -0.6, "pressure": 1.1, "rpm": -0.9, "current": 0.3, "vibration": 0.82}
          ],
          "labels": [0, 1, 2, 1]
        }' | python3 -m json.tool

echo
echo "== 3. 训练后预测（新旧字段混合均可）=="
curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1, "vibration": 0.4}
          ]
        }' | python3 -m json.tool

echo
echo "== 4. 持久化续训后的检查点 =="
curl -s -X POST "${BASE_URL}/save" \
    -H "Content-Type: application/json" \
    -d '{"path": "/app/checkpoints/model_v2.pt"}' | python3 -m json.tool
