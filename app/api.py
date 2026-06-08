import json
import os
import pickle
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pyvi.ViTokenizer import tokenize as pyvi_tokenize
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer
from underthesea import word_tokenize


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_project_path(value: Optional[str], default_relative_path: str) -> Path:
    """
    Resolve path robustly.

    If value is absolute:
        use it directly.
    If value is relative:
        resolve relative to project root, not current working directory.

    This prevents bugs when uvicorn is launched from a different cwd.
    """
    if value is None or str(value).strip() == "":
        path = ROOT_DIR / default_relative_path
    else:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT_DIR / path

    return path.resolve()


ARTIFACT_DIR = resolve_project_path(
    os.getenv("ARTIFACT_DIR"),
    "retrieval_artifacts",
)

INDEX_PATH = resolve_project_path(
    os.getenv("INDEX_PATH"),
    str(ARTIFACT_DIR / "news.index"),
)

METADATA_PATH = resolve_project_path(
    os.getenv("METADATA_PATH"),
    str(ARTIFACT_DIR / "metadata.jsonl"),
)

BM25_PATH = resolve_project_path(
    os.getenv("BM25_PATH"),
    str(ARTIFACT_DIR / "bm25.pkl"),
)

CONFIG_PATH = resolve_project_path(
    os.getenv("CONFIG_PATH"),
    str(ARTIFACT_DIR / "config.json"),
)

MODEL_DIR = resolve_project_path(
    os.getenv("MODEL_DIR"),
    "models",
)

NER_MODEL_DIR = resolve_project_path(
    os.getenv("NER_MODEL_DIR"),
    str(MODEL_DIR / "PhoBERT"),
)

EMBEDDING_MODEL_DIR = resolve_project_path(
    os.getenv("EMBEDDING_MODEL_DIR"),
    str(MODEL_DIR / "Vietnamese_Embedding"),
)

VNCORENLP_DIR = resolve_project_path(
    os.getenv("VNCORENLP_DIR"),
    str(MODEL_DIR / "VnCoreNLP"),
)

DEFAULT_SEGMENTER = os.getenv("DEFAULT_SEGMENTER", "underthesea").lower().strip()
SUPPORTED_SEGMENTERS = {"underthesea", "pyvi", "vncorenlp"}


def resolve_device() -> str:
    requested_device = os.getenv("DEVICE", "cpu").lower().strip()

    if requested_device == "cpu":
        return "cpu"

    if requested_device in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested_device == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return "cpu"


DEVICE = resolve_device()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class AppState:
    index = None
    bm25 = None
    metadata: List[Dict[str, Any]] = []
    config: Dict[str, Any] = {}
    embedder = None
    ner_tokenizer = None
    ner_model = None
    id2label = None
    vncorenlp_segmenter = None
    vncorenlp_error: Optional[str] = None


state = AppState()
model_lock = threading.Lock()
segmenter_lock = threading.Lock()


app = FastAPI(
    title="Vietnamese News Entity Retrieval API",
    version="1.2.0",
    description="PhoBERT NER + Vietnamese Embedding + FAISS + BM25 + selectable word segmentation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int = Field(default=256, ge=16, le=512)
    segmenter: str = Field(default=DEFAULT_SEGMENTER)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: str = Field(default="hybrid")
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    segmenter: str = Field(default=DEFAULT_SEGMENTER)


class EntitySearchRequest(BaseModel):
    entity_text: str = Field(..., min_length=1)
    original_text: Optional[str] = None
    method: str = Field(default="hybrid")
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    segmenter: str = Field(default=DEFAULT_SEGMENTER)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def normalize_segmenter(segmenter: Optional[str]) -> str:
    value = (segmenter or DEFAULT_SEGMENTER).lower().strip()

    aliases = {
        "underthesea": "underthesea",
        "uts": "underthesea",
        "pyvi": "pyvi",
        "vncorenlp": "vncorenlp",
        "vncore": "vncorenlp",
        "vnc": "vncorenlp",
    }

    value = aliases.get(value, value)

    if value not in SUPPORTED_SEGMENTERS:
        raise ValueError(
            f"Unsupported segmenter: {segmenter}. "
            f"Use one of: {sorted(SUPPORTED_SEGMENTERS)}"
        )

    return value


def vncorenlp_required_paths() -> Dict[str, Path]:
    return {
        "jar": VNCORENLP_DIR / "VnCoreNLP-1.2.jar",
        "wordsegmenter": VNCORENLP_DIR / "models" / "wordsegmenter" / "wordsegmenter.rdr",
        "vocab": VNCORENLP_DIR / "models" / "wordsegmenter" / "vi-vocab",
    }


def vncorenlp_file_status() -> Dict[str, Any]:
    paths = vncorenlp_required_paths()

    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "size": path.stat().st_size if path.exists() and path.is_file() else None,
        }
        for key, path in paths.items()
    }


def validate_vncorenlp_files() -> None:
    status = vncorenlp_file_status()
    missing = []

    for key, info in status.items():
        if not info["exists"] or not info["is_file"]:
            missing.append(f"{key}: {info['path']}")

    if missing:
        detail = "Missing VnCoreNLP files:\n" + "\n".join(missing)
        detail += f"\nVNCORENLP_DIR={VNCORENLP_DIR}"
        detail += f"\nROOT_DIR={ROOT_DIR}"
        raise FileNotFoundError(detail)


def initialize_vncorenlp() -> None:
    """
    Initialize VnCoreNLP once during FastAPI startup.

    This is intentionally NOT lazy-loaded. pyjnius/JVM should be started once
    before request handling begins; starting it inside concurrent requests can
    cause: "VM is already running, can't set classpath/options".

    If initialization fails, the API keeps running and only the vncorenlp
    segmenter returns HTTP 503. underthesea and pyvi still work.
    """
    try:
        validate_vncorenlp_files()

        import py_vncorenlp

        save_dir = str(VNCORENLP_DIR.resolve())

        print("=" * 100)
        print("Initializing VnCoreNLP at startup")
        print(f"ROOT_DIR: {ROOT_DIR}")
        print(f"VNCORENLP_DIR: {VNCORENLP_DIR}")
        print(f"save_dir: {save_dir}")
        print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
        print(f"JVM_PATH: {os.environ.get('JVM_PATH')}")
        print(f"CLASSPATH: {os.environ.get('CLASSPATH')}")
        print(f"File status: {json.dumps(vncorenlp_file_status(), ensure_ascii=False, indent=2)}")
        print("=" * 100)

        state.vncorenlp_segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=save_dir,
        )
        state.vncorenlp_error = None

        print("VnCoreNLP word segmenter loaded.")

    except Exception as exc:
        state.vncorenlp_segmenter = None
        state.vncorenlp_error = repr(exc)
        print(f"Failed to initialize VnCoreNLP. It will be unavailable. Error: {repr(exc)}")


def get_vncorenlp_segmenter():
    if state.vncorenlp_segmenter is not None:
        return state.vncorenlp_segmenter

    detail = state.vncorenlp_error or "VnCoreNLP was not initialized during startup."
    raise RuntimeError(f"VnCoreNLP segmenter is not initialized. Detail: {detail}")


def word_segment_text(text: str, segmenter: Optional[str] = None) -> str:
    text = normalize_text(text)

    if not text:
        return ""

    selected = normalize_segmenter(segmenter)

    if selected == "underthesea":
        return word_tokenize(text, format="text")

    if selected == "pyvi":
        return pyvi_tokenize(text)

    if selected == "vncorenlp":
        segmenter_obj = get_vncorenlp_segmenter()

        with segmenter_lock:
            segmented_sentences = segmenter_obj.word_segment(text)

        return " ".join(segmented_sentences)

    raise ValueError(f"Unsupported segmenter: {selected}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def ensure_file_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def extract_entities_from_bio(words: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    entities = []
    current_tokens = []
    current_type = None
    start_idx = None

    def flush(end_idx: int) -> None:
        nonlocal current_tokens, current_type, start_idx

        if current_tokens:
            segmented_text = " ".join(current_tokens)
            display_text = segmented_text.replace("_", " ")

            entities.append({
                "text": normalize_text(display_text),
                "segmented_text": normalize_text(segmented_text),
                "type": current_type,
                "start_word": start_idx,
                "end_word": end_idx,
            })

        current_tokens = []
        current_type = None
        start_idx = None

    for i, (word, tag) in enumerate(zip(words, tags)):
        if tag == "O" or tag is None:
            flush(i)
            continue

        if "-" not in tag:
            flush(i)
            continue

        prefix, ent_type = tag.split("-", 1)

        if prefix == "B":
            flush(i)
            current_tokens = [word]
            current_type = ent_type
            start_idx = i

        elif prefix == "I":
            if current_tokens and current_type == ent_type:
                current_tokens.append(word)
            else:
                flush(i)
                current_tokens = [word]
                current_type = ent_type
                start_idx = i

        else:
            flush(i)

    flush(len(words))
    return entities


def get_label_from_id(label_id: int) -> str:
    id2label = state.id2label

    if isinstance(id2label, dict):
        if label_id in id2label:
            return id2label[label_id]

        str_id = str(label_id)

        if str_id in id2label:
            return id2label[str_id]

    return str(label_id)


def predict_ner_entities(text: str, max_length: int = 256, segmenter: Optional[str] = None) -> Dict[str, Any]:
    selected_segmenter = normalize_segmenter(segmenter)
    segmented_text = word_segment_text(text, selected_segmenter)
    words = segmented_text.split()

    if not words:
        return {
            "text": text,
            "segmenter": selected_segmenter,
            "segmented_text": segmented_text,
            "word_labels": [],
            "entities": [],
        }

    tokens = []
    token_to_word = []

    for word_idx, word in enumerate(words):
        sub_tokens = state.ner_tokenizer.tokenize(word)

        if len(sub_tokens) == 0:
            continue

        for sub_token in sub_tokens:
            tokens.append(sub_token)
            token_to_word.append(word_idx)

    max_token_len = max_length - 2
    tokens = tokens[:max_token_len]
    token_to_word = token_to_word[:max_token_len]

    input_ids = state.ner_tokenizer.convert_tokens_to_ids(tokens)

    cls_token_id = state.ner_tokenizer.cls_token_id
    sep_token_id = state.ner_tokenizer.sep_token_id

    if cls_token_id is None:
        cls_token_id = state.ner_tokenizer.convert_tokens_to_ids("<s>")

    if sep_token_id is None:
        sep_token_id = state.ner_tokenizer.convert_tokens_to_ids("</s>")

    input_ids = [cls_token_id] + input_ids + [sep_token_id]
    attention_mask = [1] * len(input_ids)

    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(DEVICE)

    with model_lock:
        with torch.inference_mode():
            outputs = state.ner_model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )

    pred_ids = outputs.logits.argmax(dim=-1)[0].detach().cpu().tolist()
    pred_ids_without_special = pred_ids[1:-1]

    word_level_labels = []
    seen_word_ids = set()

    for token_idx, word_idx in enumerate(token_to_word):
        if word_idx in seen_word_ids:
            continue

        if token_idx >= len(pred_ids_without_special):
            break

        seen_word_ids.add(word_idx)

        label_id = int(pred_ids_without_special[token_idx])
        label = get_label_from_id(label_id)

        word_level_labels.append({
            "word": words[word_idx],
            "label": label,
        })

    pred_words = [item["word"] for item in word_level_labels]
    pred_tags = [item["label"] for item in word_level_labels]

    entities = extract_entities_from_bio(pred_words, pred_tags)

    return {
        "text": text,
        "segmenter": selected_segmenter,
        "segmented_text": segmented_text,
        "word_labels": word_level_labels,
        "entities": entities,
    }


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return scores

    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score < 1e-8:
        return np.ones_like(scores)

    return (scores - min_score) / (max_score - min_score)


def format_result(
    doc_idx: int,
    score: float,
    method: str,
    vector_score: Optional[float] = None,
    bm25_score: Optional[float] = None,
) -> Dict[str, Any]:
    doc = state.metadata[int(doc_idx)]

    result = {
        "doc_id": doc["id"],
        "score": float(score),
        "method": method,
        "display_text": doc["display_text"],
        "segmented_text": doc["segmented_text"],
        "entities": doc.get("entities", []),
        "entity_texts": doc.get("entity_texts", []),
        "entity_types": doc.get("entity_types", []),
    }

    if vector_score is not None:
        result["vector_score"] = float(vector_score)

    if bm25_score is not None:
        result["bm25_score"] = float(bm25_score)

    return result


def vector_search(query: str, top_k: int = 5, segmenter: Optional[str] = None) -> List[Dict[str, Any]]:
    segmented_query = word_segment_text(query, segmenter)

    with model_lock:
        query_embedding = state.embedder.encode(
            [segmented_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    scores, indices = state.index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append(
            format_result(
                doc_idx=int(idx),
                score=float(score),
                method="vector",
            )
        )

    return results


def bm25_search(query: str, top_k: int = 5, segmenter: Optional[str] = None) -> List[Dict[str, Any]]:
    segmented_query = word_segment_text(query, segmenter)
    query_tokens = segmented_query.split()

    scores = state.bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append(
            format_result(
                doc_idx=int(idx),
                score=float(scores[idx]),
                method="bm25",
            )
        )

    return results


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 50,
    alpha: float = 0.6,
    segmenter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    segmented_query = word_segment_text(query, segmenter)

    with model_lock:
        query_embedding = state.embedder.encode(
            [segmented_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    candidate_k = min(candidate_k, len(state.metadata))

    vector_scores, vector_indices = state.index.search(query_embedding, candidate_k)

    query_tokens = segmented_query.split()
    bm25_scores_all = state.bm25.get_scores(query_tokens)

    candidate_indices = set(int(i) for i in vector_indices[0] if i >= 0)

    bm25_top_indices = np.argsort(bm25_scores_all)[::-1][:candidate_k]
    candidate_indices.update(int(i) for i in bm25_top_indices)

    candidate_indices = list(candidate_indices)

    vector_score_map = {
        int(idx): float(score)
        for score, idx in zip(vector_scores[0], vector_indices[0])
        if idx >= 0
    }

    vector_scores_raw = np.array(
        [vector_score_map.get(idx, 0.0) for idx in candidate_indices],
        dtype=np.float32,
    )

    bm25_scores_raw = np.array(
        [bm25_scores_all[idx] for idx in candidate_indices],
        dtype=np.float32,
    )

    vector_scores_norm = min_max_normalize(vector_scores_raw)
    bm25_scores_norm = min_max_normalize(bm25_scores_raw)

    final_scores = alpha * vector_scores_norm + (1.0 - alpha) * bm25_scores_norm

    ranked = sorted(
        zip(candidate_indices, final_scores, vector_scores_raw, bm25_scores_raw),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    results = []

    for idx, final_score, vector_score, bm25_score in ranked:
        results.append(
            format_result(
                doc_idx=int(idx),
                score=float(final_score),
                method="hybrid",
                vector_score=float(vector_score),
                bm25_score=float(bm25_score),
            )
        )

    return results


def build_entity_query(entity_text: str, original_text: Optional[str] = None) -> str:
    entity_text = normalize_text(entity_text)

    if original_text:
        original_text = normalize_text(original_text)
        return f"{entity_text}. Ngữ cảnh: {original_text}"

    return entity_text


def search_documents(
    query: str,
    method: str = "hybrid",
    top_k: int = 5,
    alpha: float = 0.6,
    segmenter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    method = method.lower().strip()

    if method == "vector":
        return vector_search(query, top_k=top_k, segmenter=segmenter)

    if method == "bm25":
        return bm25_search(query, top_k=top_k, segmenter=segmenter)

    if method == "hybrid":
        return hybrid_search(
            query=query,
            top_k=top_k,
            candidate_k=max(50, top_k * 10),
            alpha=alpha,
            segmenter=segmenter,
        )

    raise ValueError("method must be one of: bm25, vector, hybrid")


@app.on_event("startup")
def startup_event() -> None:
    try:
        print("=" * 100)
        print("API startup")
        print(f"ROOT_DIR: {ROOT_DIR}")
        print(f"DEVICE: {DEVICE}")
        print(f"ARTIFACT_DIR: {ARTIFACT_DIR}")
        print(f"INDEX_PATH: {INDEX_PATH}")
        print(f"METADATA_PATH: {METADATA_PATH}")
        print(f"BM25_PATH: {BM25_PATH}")
        print(f"CONFIG_PATH: {CONFIG_PATH}")
        print(f"MODEL_DIR: {MODEL_DIR}")
        print(f"NER_MODEL_DIR: {NER_MODEL_DIR}")
        print(f"EMBEDDING_MODEL_DIR: {EMBEDDING_MODEL_DIR}")
        print(f"VNCORENLP_DIR: {VNCORENLP_DIR}")
        print("=" * 100)

        ensure_file_exists(INDEX_PATH, "FAISS index")
        ensure_file_exists(METADATA_PATH, "metadata.jsonl")
        ensure_file_exists(BM25_PATH, "bm25.pkl")
        ensure_file_exists(NER_MODEL_DIR, "PhoBERT model directory")
        ensure_file_exists(EMBEDDING_MODEL_DIR, "Vietnamese embedding model directory")

        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                state.config = json.load(f)
        else:
            state.config = {}

        print(f"Loading FAISS index: {INDEX_PATH}")
        state.index = faiss.read_index(str(INDEX_PATH))

        print(f"Loading metadata: {METADATA_PATH}")
        state.metadata = load_jsonl(METADATA_PATH)

        print(f"Loading BM25: {BM25_PATH}")
        with BM25_PATH.open("rb") as f:
            state.bm25 = pickle.load(f)

        print(f"Loading embedding model: {EMBEDDING_MODEL_DIR}")
        state.embedder = SentenceTransformer(str(EMBEDDING_MODEL_DIR), device=DEVICE)

        print(f"Loading PhoBERT tokenizer/model: {NER_MODEL_DIR}")
        state.ner_tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_DIR), use_fast=False)
        state.ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_DIR))
        state.ner_model.to(DEVICE)
        state.ner_model.eval()
        state.id2label = state.ner_model.config.id2label

        # Initialize VnCoreNLP once at startup to avoid pyjnius/JVM race conditions
        # during concurrent requests. If this fails, only the vncorenlp segmenter
        # is unavailable; underthesea and pyvi still work.
        initialize_vncorenlp()

        print("Application startup complete.")

    except Exception as exc:
        raise RuntimeError(f"Failed to initialize API: {exc}") from exc


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Vietnamese News Entity Retrieval API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE,
        "device_env": os.getenv("DEVICE", "cpu"),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ),
        "num_docs": len(state.metadata),
        "faiss_ntotal": int(state.index.ntotal) if state.index is not None else 0,
        "artifact_dir": str(ARTIFACT_DIR),
        "model_dir": str(MODEL_DIR),
        "ner_model_dir": str(NER_MODEL_DIR),
        "embedding_model_dir": str(EMBEDDING_MODEL_DIR),
        "vncorenlp_dir": str(VNCORENLP_DIR),
        "segmenters": {
            "underthesea": True,
            "pyvi": True,
            "vncorenlp": state.vncorenlp_segmenter is not None,
        },
        "vncorenlp_loaded": state.vncorenlp_segmenter is not None,
        "vncorenlp_error": state.vncorenlp_error,
        "vncorenlp_files": vncorenlp_file_status(),
        "default_segmenter": DEFAULT_SEGMENTER,
    }


@app.get("/debug/paths")
def debug_paths() -> Dict[str, Any]:
    return {
        "root_dir": str(ROOT_DIR),
        "cwd": os.getcwd(),
        "env": {
            "ARTIFACT_DIR": os.getenv("ARTIFACT_DIR"),
            "MODEL_DIR": os.getenv("MODEL_DIR"),
            "NER_MODEL_DIR": os.getenv("NER_MODEL_DIR"),
            "EMBEDDING_MODEL_DIR": os.getenv("EMBEDDING_MODEL_DIR"),
            "VNCORENLP_DIR": os.getenv("VNCORENLP_DIR"),
            "JAVA_HOME": os.getenv("JAVA_HOME"),
            "JVM_PATH": os.getenv("JVM_PATH"),
            "CLASSPATH": os.getenv("CLASSPATH"),
        },
        "resolved_paths": {
            "artifact_dir": str(ARTIFACT_DIR),
            "index_path": str(INDEX_PATH),
            "metadata_path": str(METADATA_PATH),
            "bm25_path": str(BM25_PATH),
            "config_path": str(CONFIG_PATH),
            "model_dir": str(MODEL_DIR),
            "ner_model_dir": str(NER_MODEL_DIR),
            "embedding_model_dir": str(EMBEDDING_MODEL_DIR),
            "vncorenlp_dir": str(VNCORENLP_DIR),
        },
        "vncorenlp_files": vncorenlp_file_status(),
    }


@app.post("/segment")
def segment_text(payload: NERRequest) -> Dict[str, Any]:
    try:
        selected_segmenter = normalize_segmenter(payload.segmenter)
        segmented_text = word_segment_text(payload.text, selected_segmenter)

        return {
            "text": payload.text,
            "segmenter": selected_segmenter,
            "segmented_text": segmented_text,
            "tokens": segmented_text.split(),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ner")
def ner(payload: NERRequest) -> Dict[str, Any]:
    try:
        return predict_ner_entities(
            payload.text,
            max_length=payload.max_length,
            segmenter=payload.segmenter,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search")
def search(payload: SearchRequest) -> Dict[str, Any]:
    try:
        selected_segmenter = normalize_segmenter(payload.segmenter)

        results = search_documents(
            query=payload.query,
            method=payload.method,
            top_k=payload.top_k,
            alpha=payload.alpha,
            segmenter=selected_segmenter,
        )

        return {
            "query": payload.query,
            "segmenter": selected_segmenter,
            "segmented_query": word_segment_text(payload.query, selected_segmenter),
            "method": payload.method,
            "top_k": payload.top_k,
            "alpha": payload.alpha,
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/entity-search")
def entity_search(payload: EntitySearchRequest) -> Dict[str, Any]:
    try:
        selected_segmenter = normalize_segmenter(payload.segmenter)
        final_query = build_entity_query(payload.entity_text, payload.original_text)

        results = search_documents(
            query=final_query,
            method=payload.method,
            top_k=payload.top_k,
            alpha=payload.alpha,
            segmenter=selected_segmenter,
        )

        return {
            "entity_text": payload.entity_text,
            "original_text": payload.original_text,
            "final_query": final_query,
            "segmenter": selected_segmenter,
            "segmented_query": word_segment_text(final_query, selected_segmenter),
            "method": payload.method,
            "top_k": payload.top_k,
            "alpha": payload.alpha,
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc