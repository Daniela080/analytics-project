# analytics-project

Proyecto del semestre — Analítica de Datos.

## Setup

```bash
uv sync --all-groups
uv run pytest
```

## Módulos

- `src/analytics/extract.py` — ingesta y joins verificados de Olist
- `src/analytics/quality.py` — contratos de calidad (pandera)
- `src/analytics/transform.py` / `eda.py` — features sin fuga temporal y EDA
- `src/analytics/train.py` / `evaluate.py` — pipelines supervisados y métricas
- `src/analytics/segment.py` — segmentación RFM con K-Means
- `src/analytics/pipeline.py` — orquestación completa y artefactos en `models/`
- `src/analytics/api.py` — API REST (FastAPI) para servir los modelos

## Pipeline y API

Los CSV de Olist van en `data/raw/` (no se versionan).

```bash
uv run python -m analytics.pipeline --data-dir data/raw --models-dir models
uv run uvicorn analytics.api:crear_app --factory --port 8000
```

Documentación interactiva en http://localhost:8000/docs
