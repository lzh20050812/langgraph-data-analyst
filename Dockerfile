# ============================================================
# AI Data Analyst — 多智能体企业智能运营分析平台
#
# 基于 LangGraph + FastAPI + MySQL + ECharts
# ============================================================

FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src ./src
RUN npm run build


FROM python:3.11-slim-bookworm

# 系统依赖
RUN apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 先装依赖（利用 Docker 缓存层）
COPY requirements.txt .
# The application is CPU-only. Installing torch from PyPI now resolves a CUDA
# toolchain on Linux, which makes the image unnecessarily large and can exhaust
# Docker Desktop storage during a clean build.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# 创建数据目录（含模型缓存目录，通过 volume 持久化避免每次重启重下载）
RUN mkdir -p /app/data/raw /app/data/processed /app/data/chromadb \
    /app/data/hf_cache /app/data/st_cache

# 暴露 FastAPI 端口
EXPOSE 8000

# 环境变量默认值（可通过 docker-compose 覆盖）
ENV MYSQL_HOST=mysql
ENV MYSQL_PORT=3306
ENV MYSQL_USER=root
ENV MYSQL_PASSWORD=analytics_dev_password
ENV MYSQL_DATABASE=ai_analytics
ENV CHROMA_PERSIST_DIR=/app/data/chromadb
# 模型缓存持久化（避免每次容器重启重新下载 BAAI/bge-small-zh）
ENV HF_HOME=/app/data/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/st_cache

# 启动命令
CMD ["sh", "-c", "python -m storage.init_storage --skip-etl && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
