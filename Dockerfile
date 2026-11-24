# CPU 版 PyTorch 单作业 API 镜像
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_NUM_THREADS=1

WORKDIR /app

# 先装依赖（利用层缓存）；torch 走 CPU wheel 源
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu \
    && pip install --no-cache-dir \
        "fastapi==0.165.0" \
        "uvicorn[standard]==0.54.0" \
        "pydantic==2.14.4" \
        "numpy==2.4.6"

COPY app ./app
COPY checkpoints ./checkpoints
COPY scripts ./scripts

# 检查点目录挂载为卷，容器内保存的新版本可在宿主机保留
VOLUME ["/app/checkpoints"]

EXPOSE 8000

# 单 worker：进程内只保留一个活动模型（单作业），避免多进程状态分叉
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
