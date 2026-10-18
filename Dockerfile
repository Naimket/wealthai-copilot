# Two images from one file:  docker build --target api .   |   docker build --target ui .
# Data (KID PDFs, Qdrant index, portfolio database) is NOT baked in: it is mounted at /app/data.

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# ---- API ------------------------------------------------------------------------------------------------------
FROM base AS api
RUN pip install --no-cache-dir uv
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src FASTEMBED_CACHE_PATH=/app/data/fastembed_cache
EXPOSE 8000
CMD ["uvicorn", "wealthai.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- UI (needs only Streamlit and httpx: it talks to the API over HTTP) ------------------------------------------
FROM base AS ui
RUN pip install --no-cache-dir streamlit==1.66.0 httpx
COPY app ./app
ENV WEALTHAI_API_URL=http://api:8000
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
