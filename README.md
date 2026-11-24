# 设备振动测点扩列续训 API

产线加装振动测点后，分类模型可在**不停机、不重训**的前提下追加输入列并继续训练；
未改造设备仍用原有字段预测，缺失的新列自动补 0。基于 **FastAPI + CPU 版 PyTorch**
实现的单作业（single-job）API，附带已训练的两层分类模型及**含 Adam 状态**的检查点。

面向内存紧张的巡检节点，启动与 `/reload` 采用**延迟物化**：只读入权重、列名与
Adam 步数供 `/predict`、`/model` 使用；动量张量（`exp_avg`/`exp_avg_sq`）以 mmap
留在检查点文件中，首次 `/train`、`/expand` 或 `/save` 时才读入并转换精度。

## 1. 目录结构

```
.
├── app/
│   ├── model.py          # 两层分类器与扩列（零权重追加列）
│   ├── checkpoint.py     # 检查点加载（含延迟物化）/扩列/原子保存（含 Adam 状态迁移）
│   ├── data.py           # 按列名对齐、缺列补 0、多余列忽略
│   ├── schemas.py        # 请求/响应模型
│   ├── service.py        # 单作业状态、训练一步、预测、保存、重载
│   └── main.py           # FastAPI 路由与启动加载
├── scripts/
│   └── train_initial.py  # 生成初始检查点（合成数据，200 步 Adam）
├── tests/
│   ├── test_core.py      # 数值对照 / 扩列 / Adam 状态 / 延迟物化等 16 项
│   └── test_api.py       # HTTP 端到端 10 个用例（共 26 项断言场景）
├── checkpoints/
│   └── model.pt          # 已训练检查点（4 输入 → 8 隐藏 → 3 类，含 Adam 状态）
├── examples/
│   ├── expand_api.sh              # 扩列 API 示例
│   ├── retrain_api.sh             # 续训 API 示例
│   ├── predict_only_api.sh        # 只做预测示例（不触发动量读取）
│   └── replace_and_retrain_api.sh # 原子替换检查点后续训示例
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
- 检查点格式版本字段 `version=1`，加载入口 `app.checkpoint.load_checkpoint`
  （`defer_optimizer=True` 时延迟物化优化器，见下节）。

## 3. 延迟物化（启动 / 重载的内存优化）

内存紧张的巡检节点上，启动与 `/reload` 只读入**权重、列名与 Adam 步数**：

| 阶段 | 读入内容 | 服务的接口 |
| --- | --- | --- |
| 启动 / `/reload` | 模型权重（拷入模型并转运行精度）、`feature_names`、Adam `step` | `/predict`、`/model` |
| 首次 `/train` / `/expand` / `/save` | 动量张量 `exp_avg`/`exp_avg_sq` 读入内存并转运行精度 | 训练、扩列、保存 |

实现要点：

- `load_checkpoint(..., defer_optimizer=True)` 以 `torch.load(mmap=True)` 打开
  检查点：动量张量以 mmap 留在文件中，未访问的页不占常驻内存；
  `CheckpointBundle.ensure_optimizer()` 在首次需要时物化 Adam 并做精度转换
  （运行精度与文件精度不同时也一样，转换同样推迟到物化时）。
- mmap 钉住了加载时的文件 inode：**原路径被原子替换后，首次取用的动量仍来自
  加载当前权重时用的那份文件**；显式 `/reload` 才切换到新文件。
  （注意：仅保证“原子替换”语义；对文件做原地改写不属于该语义。）
- `/reload` 先完整构建新模型再整体替换：**重载失败时原模型仍可预测和续训**，
  续训的动量同样来自旧模型加载时钉住的文件快照。
- 一致性：同一运行精度与输入下，延迟物化与原加载方式（`defer_optimizer=False`）
  首次续训/扩列后的权重与 Adam 状态**逐位一致**（测试误差 0）；
  `/save` 写出的仍是可按原格式加载的完整检查点。
- 老版本 PyTorch（无 `mmap` 参数）或旧格式检查点自动退回全量读取，
  数值与“同一份文件”语义不变，仅失去省内存效果。

## 4. 扩列与续训的一致性设计

| 需求 | 实现方式 |
| --- | --- |
| 按列名对齐 | `app/data.py` 以模型 `feature_names` 为准建立列索引，请求中的列名映射到固定位置 |
| 缺新列补 0 | 对齐张量先以 0 初始化，缺失列保持 0；响应中通过 `zero_filled_columns` 回显 |
| 多余列忽略 | 不在 `feature_names` 中的键直接丢弃 |
| 扩列后旧分数不变 | 仅在 `fc1.weight` **尾部追加零初始化列**，偏置与 `fc2` 不变；零列对补 0 的旧样本前向无贡献 |
| 续训一步后原参数/Adam 一致 | 旧参数及其 `exp_avg`/`exp_avg_sq`/`step` 原样迁移；新列动量同样以 0 追加，参数顺序不变 |
| 新读数更新新权重 | 带 `vibration` 的样本使新列产生非零梯度，Adam 一步即可更新新增权重 |
| 确定性 | 单线程 CPU（`TORCH_NUM_THREADS=1`）+ `torch.use_deterministic_algorithms(True)` |

## 5. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/model` | 当前列名、维度、dtype、Adam step（不触发动量读取） |
| POST | `/predict` | 预测；缺列补 0，返回标签、各类概率、补 0 列 |
| POST | `/train` | 用一批样本**训练一步**并返回训练前后 loss（首次调用物化动量） |
| POST | `/expand` | 追加一个或多个测点列（零权重 + Adam 零状态） |
| POST | `/save` | 原子保存检查点（先写临时文件再 `os.replace`） |
| POST | `/reload` | 从磁盘重载（模拟重启/回退），支持 query 参数 `path` |

训练、扩列、保存、重载由同一把异步锁串行化；预测为只读，不阻塞。

### 保存与回退语义

- 保存走“临时文件 + 原子 rename”，**保存失败时原检查点与内存模型均不受影响**，服务继续可用。
- 保存新版本（如 `model_v2.pt`）后，旧文件 `model.pt` 保持不变，仍可通过 `/reload` 切回。
- 重启容器后从 `CHECKPOINT_PATH` 加载；把环境变量指向新版本即恢复扩列后的模型，
  旧字段预测与按新列续训均可用。
- 重载先完整构建新模型再整体替换，**重载失败（文件缺失/损坏）返回 500，
  原模型仍可预测和续训**。

## 6. 本地运行（不使用 Docker）

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/train_initial.py          # （可选）重新生成 checkpoints/model.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 7. Docker Compose 启动

```bash
docker compose up -d --build
docker compose logs -f classifier-api
curl http://127.0.0.1:8000/model
```

`./checkpoints` 挂载进容器，容器内保存的新版本在宿主机同目录可见。
镜像以单 uvicorn worker 运行，保证进程内只有一个活动模型。

### 只做预测示例

```bash
bash examples/predict_only_api.sh
```

`/model` 与 `/predict` 只使用启动时读入的权重、列名与 Adam 步数，
不触发动量张量的读取，适合内存紧张节点长期只跑预测。

### 替换文件后续训示例

```bash
bash examples/replace_and_retrain_api.sh
```

演示：服务运行期间用 `mv` 原子替换 `./checkpoints/model.pt` 后，
首次 `/train` 的动量仍来自加载权重时钉住的那份文件（step 接续
加载时的值），显式 `/reload` 后才切换到替换后的文件。

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

## 8. 自测与实测结果

```bash
python -m pytest tests/ -v
```

**实测环境**：Python 3.11 / torch 2.15.0+cpu / FastAPI 0.165，float32，单线程。

需求中的两处硬性数值对照（12 条随机旧样本，全量 26 个测试通过）：

| 对照项 | 限值 | 实测最大绝对误差 |
| --- | --- | --- |
| 扩列后、续训前旧样本分类分数（softmax 概率） | 1e-6 | **0.000e+00** |
| 同批旧样本各训一步后的原参数（4 个张量） | 1e-6 | **0.000e+00** |
| 同上，对应 Adam 的 `exp_avg`/`exp_avg_sq`/`step` | 1e-6 | **0.000e+00** |
| 延迟物化 vs 原加载方式：首次续训后的权重与 Adam 状态 | 1e-6 | **0.000e+00** |
| 延迟物化 vs 原加载方式：首次扩列（含续训）后的权重与 Adam 状态 | 1e-6 | **0.000e+00** |
| 运行精度 float64 ≠ 文件精度 float32：首次续训后同上 | 1e-6 | **0.000e+00** |

延迟物化实测：

- **常驻内存**：51.2 MB 检查点（权重 16.8 MB + 动量 33.5 MB），
  启动只读权重/步数后 RSS 增量 **36.5 MB**，原方式全量加载为 **147.2 MB**
  （动量约占 2/3 的场景下，启动内存降至约 1/4）；首次训练物化动量后
  与原方式相当。
- **原子替换**：加载后用 `os.replace` 换掉原路径文件，首次续训的动量
  仍逐位等于加载时的文件内容（step 200→201，而非替换后文件的 203→204），
  与“原方式加载旧文件训一步”的结果逐位一致。
- **重载失败**：损坏/缺失文件返回 500 后，原模型预测正常、续训 step 正常
  递增；即使原路径随后被破坏，续训仍从加载时钉住的文件快照取动量。
- **保存**：懒加载后 `/save` 触发物化，写出的检查点可按原格式完整加载，
  权重与 Adam 状态同原方式逐位一致。

其他实测结论：

- 新列零初始化；仅喂旧字段训练一步后新列权重仍为 0；带 `vibration` 读数训练后新列权重变为非零。
- 保存到非法路径返回 500，原 `model.pt` 可重新加载、服务预测不受影响。
- 保存 → 重载（等价重启）→ 旧字段预测正常（补 0 列正确回显）→ 按新列续训，Adam step 正确递增（201→202→203）。
- 重名列、空扩列、越界标签、样本/标签数量不一致均返回 422。
- 示例脚本 `predict_only_api.sh` 与 `replace_and_retrain_api.sh` 已对
  本地 uvicorn 启动的服务（与容器同一 app）完整跑通，step 序列符合预期
  （200→201/202/203→回退 200→替换后续训 201→reload 203→204）。

## 9. 已实现功能与未完成项

**已实现**

- 单作业 FastAPI 服务，CPU PyTorch，启动即加载含 Adam 状态的检查点
- **启动 / `/reload` 延迟物化**：只读权重、列名、Adam 步数；动量张量
  首次 `/train`、`/expand`、`/save` 时才从 mmap 读入并转换精度
- 原子替换检查点后，首次取用的动量仍来自加载权重时的那份文件；
  重载失败后原模型仍可预测和续训
- 两层分类模型初始训练脚本与随附检查点
- 按列名对齐、缺列补 0、多余列忽略，预测响应回显补 0 列
- 追加测点列（支持一次多列、重名拒绝），零权重 + Adam 零状态迁移
- 扩列前后旧样本分数、续训一步后原参数与 Adam 状态逐位一致（误差 0 ≤ 1e-6）
- 带新读数样本可更新新增权重；检查点原子保存、失败保护、重载恢复
- Dockerfile + docker-compose（单 worker、检查点卷挂载、健康检查）
- 扩列/续训/只做预测/替换后续训 API shell 示例与 26 项自动化测试

**未完成 / 边界说明**

- 当前环境未安装 Docker，**镜像未在本机实际构建运行**；Dockerfile/compose
  未改动、仍按 CPU wheel 与单作业要求编写，需在有 Docker 的环境执行
  `docker compose up -d --build` 验证（示例与测试已在本地 uvicorn 上跑通）。
- “同一份文件”保证依赖 mmap 钉住 inode，覆盖**原子替换**（rename）场景；
  对检查点文件做原地改写（非原子）不在保证范围内。
- 训练接口固定为“每请求一步”的小批量增量训练，未实现多 epoch 调度、学习率调度器与早停。
- 未接入鉴权/限流与多作业并发；服务按单作业设计，写操作由进程内锁串行化。
- 合成数据仅用于演示与数值核对；接入真实产线数据时需替换 `scripts/train_initial.py` 的数据来源并重新标定类别。
