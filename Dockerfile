# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG UID=1000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid ${UID} app \
    && mkdir /app \
    && chown app:app /app
WORKDIR /app

COPY requirements.lock pyproject.toml ./
RUN pip install -r requirements.lock

COPY --chown=app:app . .
RUN pip install --no-deps -e .

USER app
ENTRYPOINT ["make"]
CMD ["test"]
