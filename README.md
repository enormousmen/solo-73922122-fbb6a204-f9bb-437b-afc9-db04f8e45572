# 设备振动测点扩列续训 API

产线加装振动测点后，分类模型可在**不停机、不重训**的前提下追加输入列并继续训练；
未改造设备仍用原有字段预测，缺失的新列自动补 0。基于 **FastAPI + CPU 版 PyTorch**
实现的单作业（single-job）API，附带已训练的两层分类模型及**含 Adam 状态**的检查点。

## 1. 目录结构

```
.
├── app/
│   ├── model.py          # 两层分类器与扩列（零权重追加列）
│   ├── checkpoint.py     # 检查点加载/扩列/原子保存（含 Adam 状态迁移）
│   ├── data.py           # 按列名对齐、缺列补 0、多余列忽略
│   ├── schemas.py        # 请求/响应模型
│   ├── service.py        # 单作业状态、训练一步、预测、保存、重载
│   └── main.py           # FastAPI 路由与启动加载
├── scripts/
│   └── train_initial.py  # 生成初始检查点（合成数据，200 步 Adam）
├── tests/
│   ├── test_core.py      # 数值对照 / 扩列 / Adam 状态 / 保存失败等 9 项
│   └── test_api.py       # HTTP 端到端 7 个用例（共 16 项断言场景）
├── checkpoints/
│   └── model.pt          # 已训练检查点（4 输入 → 8 隐藏 → 3 类，含 Adam 状态）
├── examples/
│   ├── expand_api.sh     # 扩列 API 示例
│   └── retrain_api.sh    # 续训 API 示例
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 2. 模型与检查点

- 模型：`Linear(4→8) → ReLU → Linear(8→3)`，原始字段
  `temp, pressure, rpm, current`，3 个分类。
- 初始检查点由 `scripts/train_initial.py` 在合成三分类数据上训练 200 步得到
  （训练集准确率约 99%），文件内同时保存：
  - 模型 `state_dict`
  - **Adam 优化器完整状态**（`exp_avg`、`exp_avg_sq`、`step`、超参）
  - 特征列名、隐藏维度、类别数、dtype、格式版本
- 检查点格式版本字段 `version=1`，加载入口 `app.checkpoint.load_checkpoint`。

## 3. 扩列与续训的一致性设计

| 需求 | 实现方式 |
| --- | --- |
| 按列名对齐 | `app/data.py` 以模型 `feature_names` 为准建立列索引，请求中的列名映射到固定位置 |
| 缺新列补 0 | 对齐张量先以 0 初始化，缺失列保持 0；响应中通过 `zero_filled_columns` 回显 |
| 多余列忽略 | 不在 `feature_names` 中的键直接丢弃 |
| 扩列后旧分数不变 | 仅在 `fc1.weight` **尾部追加零初始化列**，偏置与 `fc2` 不变；零列对补 0 的旧样本前向无贡献 |
| 续训一步后原参数/Adam 一致 | 旧参数及其 `exp_avg`/`exp_avg_sq`/`step` 原样迁移；新列动量同样以 0 追加，参数顺序不变 |
| 新读数更新新权重 | 带 `vibration` 的样本使新列产生非零梯度，Adam 一步即可更新新增权重 |
| 确定性 | 单线程 CPU（`TORCH_NUM_THREADS=1`）+ `torch.use_deterministic_algorithms(True)` |

## 4. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/model` | 当前列名、维度、dtype、Adam step |
| POST | `/predict` | 预测；缺列补 0，返回标签、各类概率、补 0 列 |
| POST | `/train` | 用一批样本**训练一步**并返回训练前后 loss |
| POST | `/expand` | 追加一个或多个测点列（零权重 + Adam 零状态） |
| POST | `/save` | 原子保存检查点（先写临时文件再 `os.replace`） |
| POST | `/reload` | 从磁盘重载（模拟重启/回退），支持 query 参数 `path` |

训练、扩列、保存、重载由同一把异步锁串行化；预测为只读，不阻塞。

### 保存与回退语义

- 保存走“临时文件 + 原子 rename”，**保存失败时原检查点与内存模型均不受影响**，服务继续可用。
- 保存新版本（如 `model_v2.pt`）后，旧文件 `model.pt` 保持不变，仍可通过 `/reload` 切回。
- 重启容器后从 `CHECKPOINT_PATH` 加载；把环境变量指向新版本即恢复扩列后的模型，
  旧字段预测与按新列续训均可用。

## 5. 本地运行（不使用 Docker）

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/train_initial.py          # （可选）重新生成 checkpoints/model.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. Docker Compose 启动

```bash
docker compose up -d --build
docker compose logs -f classifier-api
curl http://127.0.0.1:8000/model
```

`./checkpoints` 挂载进容器，容器内保存的新版本在宿主机同目录可见。
镜像以单 uvicorn worker 运行，保证进程内只有一个活动模型。

### 扩列示例

```bash
bash examples/expand_api.sh
```

核心请求：

```bash
curl -X POST http://127.0.0.1:8000/expand \
  -H "Content-Type: application/json" \
  -d '{"new_feature_names": ["vibration"]}'

# 未改造设备：只传原字段，vibration 自动补 0
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1}]}'
```

### 续训示例

```bash
bash examples/retrain_api.sh
```

核心请求（带新测点读数的样本）：

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1, "vibration": 0.25},
      {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9, "vibration": 0.51}
    ],
    "labels": [0, 1]
  }'
```

保存与重启恢复：

```bash
curl -X POST http://127.0.0.1:8000/save \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/checkpoints/model_v2.pt"}'

# 重启后将 CHECKPOINT_PATH 指向 model_v2.pt，或显式 reload
curl -X POST "http://127.0.0.1:8000/reload?path=/app/checkpoints/model_v2.pt"
```

## 7. 自测与实测结果

```bash
python -m pytest tests/ -v
```

**实测环境**：Python 3.11 / torch 2.15.0+cpu / FastAPI 0.165，float32，单线程。

需求中的两处硬性数值对照（12 条随机旧样本，全量 16 个测试通过）：

| 对照项 | 限值 | 实测最大绝对误差 |
| --- | --- | --- |
| 扩列后、续训前旧样本分类分数（softmax 概率） | 1e-6 | **0.000e+00** |
| 同批旧样本各训一步后的原参数（4 个张量） | 1e-6 | **0.000e+00** |
| 同上，对应 Adam 的 `exp_avg`/`exp_avg_sq`/`step` | 1e-6 | **0.000e+00** |

其他实测结论：

- 新列零初始化；仅喂旧字段训练一步后新列权重仍为 0；带 `vibration` 读数训练后新列权重变为非零。
- 保存到非法路径返回 500，原 `model.pt` 可重新加载、服务预测不受影响。
- 保存 → 重载（等价重启）→ 旧字段预测正常（补 0 列正确回显）→ 按新列续训，Adam step 正确递增（201→202→203）。
- 重名列、空扩列、越界标签、样本/标签数量不一致均返回 422。

## 8. 已实现功能与未完成项

**已实现**

- 单作业 FastAPI 服务，CPU PyTorch，启动即加载含 Adam 状态的检查点
- 两层分类模型初始训练脚本与随附检查点
- 按列名对齐、缺列补 0、多余列忽略，预测响应回显补 0 列
- 追加测点列（支持一次多列、重名拒绝），零权重 + Adam 零状态迁移
- 扩列前后旧样本分数、续训一步后原参数与 Adam 状态逐位一致（误差 0 ≤ 1e-6）
- 带新读数样本可更新新增权重；检查点原子保存、失败保护、重载恢复
- Dockerfile + docker-compose（单 worker、检查点卷挂载、健康检查）
- 扩列/续训 API shell 示例与 16 项自动化测试

**未完成 / 边界说明**

- 当前环境未安装 Docker，**镜像未在本机实际构建运行**；Dockerfile/compose 已按 CPU wheel 与单作业要求编写，需在有 Docker 的环境执行 `docker compose up -d --build` 验证。
- 训练接口固定为“每请求一步”的小批量增量训练，未实现多 epoch 调度、学习率调度器与早停。
- 未接入鉴权/限流与多作业并发；服务按单作业设计，写操作由进程内锁串行化。
- 合成数据仅用于演示与数值核对；接入真实产线数据时需替换 `scripts/train_initial.py` 的数据来源并重新标定类别。
