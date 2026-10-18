# Dev/runtime image. The Playwright base image ships Chromium + system deps; its tag must match
# the `playwright==` pin in pyproject.toml so the driver and the bundled browser agree.
FROM mcr.microsoft.com/playwright/python:v1.63.0-noble

COPY --from=ghcr.io/astral-sh/uv:0.9.29 /uv /uvx /usr/local/bin/

# The venv lives outside /app so a bind-mounted checkout (with a Windows .venv) can't clobber it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    BU_SESSIONS_DIR=/data/sessions

WORKDIR /app
RUN mkdir -p /opt/venv /data/sessions && chown -R pwuser:pwuser /opt/venv /data /app
USER pwuser

COPY --chown=pwuser:pwuser pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY --chown=pwuser:pwuser . .
RUN uv sync --frozen

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz').status != 200)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
