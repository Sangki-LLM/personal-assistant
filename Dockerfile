FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true && \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list 2>/dev/null || true && \
    uv run playwright install --with-deps chromium

COPY app/ ./app/
COPY app/templates/ ./app/templates/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
