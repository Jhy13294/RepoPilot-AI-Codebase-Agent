# RepoPilot runtime image for the API, Streamlit console, and stdio MCP server.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# git is required by apply_patch (git apply) and repo_manager
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Install locked runtime dependencies before copying project sources for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra console --extra mcp --no-install-project

COPY . .
RUN uv sync --frozen --no-dev --extra console --extra mcp \
    && mkdir -p /app/data

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
