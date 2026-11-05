# GPU image: the `gpu` extra pulls CUDA/cuDNN as pip wheels, so the base only needs Python.
# Run with --gpus all (requires the NVIDIA Container Toolkit on the host).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /home/triplet_loss_playground

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    KERAS_HOME=/home/triplet_loss_playground/.keras

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --extra gpu
