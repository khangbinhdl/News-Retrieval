.PHONY: install-api install-ui download-models run-api run-api-cpu run-api-gpu run-api-mps run-ui

PYTHON ?= python
PIP ?= pip

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
UI_PORT ?= 8501

DEVICE ?= cpu
LOG_LEVEL ?= debug

ARTIFACT_DIR ?= retrieval_artifacts
MODEL_DIR ?= models
NER_MODEL_DIR ?= $(MODEL_DIR)/PhoBERT
EMBEDDING_MODEL_DIR ?= $(MODEL_DIR)/Vietnamese_Embedding
VNCORENLP_DIR ?= $(MODEL_DIR)/VnCoreNLP
DEFAULT_SEGMENTER ?= underthesea

install-api:
	$(PIP) install -r requirements-api.txt

install-ui:
	$(PIP) install -r requirements-ui.txt

download-models:
	MODEL_DIR=$(MODEL_DIR) \
	NER_MODEL_DIR=$(NER_MODEL_DIR) \
	EMBEDDING_MODEL_DIR=$(EMBEDDING_MODEL_DIR) \
	VNCORENLP_DIR=$(VNCORENLP_DIR) \
	ARTIFACT_DIR=$(ARTIFACT_DIR) \
	$(PYTHON) scripts/download_models.py

run-api:
	ARTIFACT_DIR=$(ARTIFACT_DIR) \
	MODEL_DIR=$(MODEL_DIR) \
	NER_MODEL_DIR=$(NER_MODEL_DIR) \
	EMBEDDING_MODEL_DIR=$(EMBEDDING_MODEL_DIR) \
	VNCORENLP_DIR=$(VNCORENLP_DIR) \
	DEFAULT_SEGMENTER=$(DEFAULT_SEGMENTER) \
	DEVICE=$(DEVICE) \
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	VECLIB_MAXIMUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 \
	NUMBA_NUM_THREADS=1 \
	TOKENIZERS_PARALLELISM=false \
	PYTHONFAULTHANDLER=1 \
	uvicorn app.api:app --host $(API_HOST) --port $(API_PORT) --workers 1 --log-level $(LOG_LEVEL)

run-api-cpu:
	$(MAKE) run-api DEVICE=cpu

run-api-gpu:
	$(MAKE) run-api DEVICE=auto

run-api-mps:
	$(MAKE) run-api DEVICE=mps

run-ui:
	API_URL=http://$(API_HOST):$(API_PORT) \
	streamlit run app/ui.py --server.port $(UI_PORT)
