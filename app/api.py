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

RRF_K = float(os.getenv("RRF_K", "60"))
ENTITY_RERANK_MIN_CANDIDATES = int(os.getenv("ENTITY_RERANK_MIN_CANDIDATES", "50"))
ENTITY_RERANK_MAX_CANDIDATES = int(os.getenv("ENTITY_RERANK_MAX_CANDIDATES", "200"))
ENTITY_EXACT_TEXT_BONUS = float(os.getenv("ENTITY_EXACT_TEXT_BONUS", "0.35"))
ENTITY_PARTIAL_TEXT_BONUS = float(os.getenv("ENTITY_PARTIAL_TEXT_BONUS", "0.15"))
ENTITY_TYPE_BONUS = float(os.getenv("ENTITY_TYPE_BONUS", "0.10"))
ENTITY_TYPE_ONLY_BONUS = float(os.getenv("ENTITY_TYPE_ONLY_BONUS", "0.03"))


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
    artifact_validation: Dict[str, Any] = {}


state = AppState()
model_lock = threading.Lock()
segmenter_lock = threading.Lock()


app = FastAPI(
    title="Vietnamese News Entity Retrieval API",
    version="1.3.0",
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
    entity_type: Optional[str] = None
    original_text: Optional[str] = None
    method: str = Field(default="hybrid")
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    segmenter: str = Field(default=DEFAULT_SEGMENTER)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def normalize_for_match(value: Optional[str]) -> str:
    if value is None:
        return ""

    normalized = normalize_text(str(value)).replace("_", " ").casefold()
    return " ".join(normalized.split())


def normalize_entity_type(value: Optional[str]) -> str:
    if value is None:
        return ""

    normalized = normalize_text(str(value)).casefold()
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace("-", "_").replace(" ", "_")

    aliases = {
        "person_name": "name",
        "occupation": "job",
        "symptomanddisease": "symptom_and_disease",
        "symptom_and_disease": "symptom_and_disease",
        "symptom_disease": "symptom_and_disease",
    }

    return aliases.get(normalized, normalized)


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


def bm25_corpus_size(bm25: Any) -> Optional[int]:
    corpus_size = getattr(bm25, "corpus_size", None)

    if corpus_size is not None:
        return int(corpus_size)

    doc_freqs = getattr(bm25, "doc_freqs", None)

    if doc_freqs is not None:
        return len(doc_freqs)

    return None


def validate_retrieval_artifacts() -> None:
    errors = []
    warnings = []

    if state.index is None:
        errors.append("FAISS index is not loaded.")

    if state.bm25 is None:
        errors.append("BM25 object is not loaded.")

    if not state.metadata:
        errors.append("metadata.jsonl is empty.")

    metadata_count = len(state.metadata)
    index_count = int(state.index.ntotal) if state.index is not None else None
    index_dimension = int(state.index.d) if state.index is not None and hasattr(state.index, "d") else None
    bm25_count = bm25_corpus_size(state.bm25) if state.bm25 is not None else None

    if index_count is not None and index_count != metadata_count:
        errors.append(
            f"FAISS index size ({index_count}) does not match metadata rows ({metadata_count})."
        )

    if bm25_count is not None and bm25_count != metadata_count:
        errors.append(
            f"BM25 corpus size ({bm25_count}) does not match metadata rows ({metadata_count})."
        )

    if state.config:
        expected_docs = state.config.get("num_docs")
        expected_dimension = state.config.get("dimension")

        if expected_docs is not None:
            try:
                expected_docs_int = int(expected_docs)
            except (TypeError, ValueError):
                errors.append(f"config.num_docs is not an integer: {expected_docs!r}.")
            else:
                if expected_docs_int != metadata_count:
                    errors.append(
                        f"config.num_docs ({expected_docs}) does not match metadata rows ({metadata_count})."
                    )

        if expected_dimension is not None and index_dimension is not None:
            try:
                expected_dimension_int = int(expected_dimension)
            except (TypeError, ValueError):
                errors.append(f"config.dimension is not an integer: {expected_dimension!r}.")
            else:
                if expected_dimension_int != index_dimension:
                    errors.append(
                        f"config.dimension ({expected_dimension}) does not match FAISS dimension ({index_dimension})."
                    )
    else:
        warnings.append("config.json is empty.")

    required_fields = {
        "id": str,
        "display_text": str,
        "segmented_text": str,
        "entities": list,
    }
    doc_ids = []

    for row_idx, doc in enumerate(state.metadata):
        if not isinstance(doc, dict):
            errors.append(f"metadata row {row_idx} is not a JSON object.")
            continue

        for field_name, expected_type in required_fields.items():
            if field_name not in doc:
                errors.append(f"metadata row {row_idx} is missing required field '{field_name}'.")
                continue

            if not isinstance(doc[field_name], expected_type):
                errors.append(
                    f"metadata row {row_idx} field '{field_name}' has type "
                    f"{type(doc[field_name]).__name__}, expected {expected_type.__name__}."
                )

        doc_id = doc.get("id")

        if isinstance(doc_id, str):
            doc_ids.append(doc_id)

        if len(errors) >= 20:
            break

    duplicate_count = len(doc_ids) - len(set(doc_ids))

    if duplicate_count > 0:
        errors.append(f"metadata contains {duplicate_count} duplicate doc id(s).")

    state.artifact_validation = {
        "metadata_rows": metadata_count,
        "faiss_ntotal": index_count,
        "faiss_dimension": index_dimension,
        "bm25_corpus_size": bm25_count,
        "config_num_docs": state.config.get("num_docs") if state.config else None,
        "config_dimension": state.config.get("dimension") if state.config else None,
        "warnings": warnings,
        "valid": len(errors) == 0,
    }

    if errors:
        state.artifact_validation["errors"] = errors
        raise ValueError("Invalid retrieval artifacts:\n- " + "\n- ".join(errors))


def validate_embedding_model() -> None:
    if state.embedder is None or state.index is None:
        return

    embedding_dimension = state.embedder.get_sentence_embedding_dimension()

    if embedding_dimension is None:
        state.artifact_validation["embedding_dimension"] = None
        state.artifact_validation.setdefault("warnings", []).append(
            "Embedding model did not report a sentence embedding dimension."
        )
        return

    embedding_dimension = int(embedding_dimension)
    index_dimension = int(state.index.d)
    state.artifact_validation["embedding_dimension"] = embedding_dimension

    if embedding_dimension != index_dimension:
        state.artifact_validation["valid"] = False
        raise ValueError(
            f"Embedding dimension ({embedding_dimension}) does not match FAISS dimension ({index_dimension})."
        )


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


def normalize_result_scores(scores: List[float]) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float32)

    if len(scores_array) == 0:
        return scores_array

    min_score = scores_array.min()
    max_score = scores_array.max()

    if max_score - min_score < 1e-8:
        return np.zeros_like(scores_array)

    return (scores_array - min_score) / (max_score - min_score)


def reciprocal_rank(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0

    return 1.0 / (RRF_K + float(rank))


def format_result(
    doc_idx: int,
    score: float,
    method: str,
    vector_score: Optional[float] = None,
    bm25_score: Optional[float] = None,
    vector_rank: Optional[int] = None,
    bm25_rank: Optional[int] = None,
    score_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    doc = state.metadata[int(doc_idx)]

    result = {
        "doc_index": int(doc_idx),
        "doc_id": doc["id"],
        "score": float(score),
        "method": method,
        "display_text": doc["display_text"],
        "segmented_text": doc["segmented_text"],
        "entities": doc.get("entities", []),
        "entity_texts": doc.get("entity_texts", []),
        "entity_segmented_texts": doc.get("entity_segmented_texts", []),
        "entity_types": doc.get("entity_types", []),
    }

    if vector_score is not None:
        result["vector_score"] = float(vector_score)

    if bm25_score is not None:
        result["bm25_score"] = float(bm25_score)

    if vector_rank is not None:
        result["vector_rank"] = int(vector_rank)

    if bm25_rank is not None:
        result["bm25_rank"] = int(bm25_rank)

    if score_details is not None:
        result["score_details"] = score_details

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

    vector_score_map = {}
    vector_rank_map = {}

    for rank, (score, idx) in enumerate(zip(vector_scores[0], vector_indices[0]), start=1):
        if idx < 0:
            continue

        doc_idx = int(idx)
        vector_score_map[doc_idx] = float(score)
        vector_rank_map[doc_idx] = rank

    bm25_rank_map = {
        int(idx): rank
        for rank, idx in enumerate(bm25_top_indices, start=1)
    }

    ranked_items = []

    for idx in candidate_indices:
        vector_rank = vector_rank_map.get(idx)
        bm25_rank = bm25_rank_map.get(idx)
        vector_rrf = reciprocal_rank(vector_rank)
        bm25_rrf = reciprocal_rank(bm25_rank)
        final_score = alpha * vector_rrf + (1.0 - alpha) * bm25_rrf

        ranked_items.append({
            "doc_idx": idx,
            "final_score": final_score,
            "vector_score": vector_score_map.get(idx, 0.0),
            "bm25_score": float(bm25_scores_all[idx]),
            "vector_rank": vector_rank,
            "bm25_rank": bm25_rank,
            "vector_rrf": vector_rrf,
            "bm25_rrf": bm25_rrf,
        })

    ranked = sorted(
        ranked_items,
        key=lambda item: (item["final_score"], item["vector_score"], item["bm25_score"]),
        reverse=True,
    )[:top_k]

    results = []

    for item in ranked:
        results.append(
            format_result(
                doc_idx=int(item["doc_idx"]),
                score=float(item["final_score"]),
                method="hybrid",
                vector_score=float(item["vector_score"]),
                bm25_score=float(item["bm25_score"]),
                vector_rank=item["vector_rank"],
                bm25_rank=item["bm25_rank"],
                score_details={
                    "type": "weighted_rrf",
                    "alpha": float(alpha),
                    "rrf_k": float(RRF_K),
                    "vector_rrf": float(item["vector_rrf"]),
                    "bm25_rrf": float(item["bm25_rrf"]),
                },
            )
        )

    return results


def build_entity_query(entity_text: str, original_text: Optional[str] = None) -> str:
    entity_text = normalize_text(entity_text)

    if original_text:
        original_text = normalize_text(original_text)
        return f"{entity_text}. Ngữ cảnh: {original_text}"

    return entity_text


def collect_doc_entity_values(doc: Dict[str, Any], field_name: str) -> List[str]:
    values = [
        str(value)
        for value in doc.get(field_name, []) or []
        if value is not None
    ]

    entity_field_map = {
        "entity_texts": "text",
        "entity_segmented_texts": "segmented_text",
        "entity_types": "type",
    }
    nested_field_name = entity_field_map.get(field_name)

    if nested_field_name:
        for entity in doc.get("entities", []) or []:
            value = entity.get(nested_field_name)

            if value is not None:
                values.append(str(value))

    return values


def entity_match_features(
    doc: Dict[str, Any],
    entity_text: str,
    entity_type: Optional[str] = None,
) -> Dict[str, Any]:
    entity_norm = normalize_for_match(entity_text)
    entity_type_norm = normalize_entity_type(entity_type)

    entity_text_values = collect_doc_entity_values(doc, "entity_texts")
    entity_segmented_values = collect_doc_entity_values(doc, "entity_segmented_texts")
    entity_type_values = collect_doc_entity_values(doc, "entity_types")

    normalized_entity_texts = {
        normalize_for_match(value)
        for value in entity_text_values + entity_segmented_values
        if normalize_for_match(value)
    }
    normalized_entity_types = {
        normalize_entity_type(value)
        for value in entity_type_values
        if normalize_entity_type(value)
    }

    display_text_norm = normalize_for_match(doc.get("display_text", ""))
    segmented_text_norm = normalize_for_match(doc.get("segmented_text", ""))

    exact_text_match = bool(entity_norm and entity_norm in normalized_entity_texts)
    partial_text_match = bool(
        entity_norm
        and not exact_text_match
        and (
            entity_norm in display_text_norm
            or entity_norm in segmented_text_norm
            or any(entity_norm in value for value in normalized_entity_texts)
            or any(value in entity_norm for value in normalized_entity_texts if value)
        )
    )
    type_match = bool(entity_type_norm and entity_type_norm in normalized_entity_types)

    text_bonus = 0.0

    if exact_text_match:
        text_bonus = ENTITY_EXACT_TEXT_BONUS
    elif partial_text_match:
        text_bonus = ENTITY_PARTIAL_TEXT_BONUS

    if type_match and (exact_text_match or partial_text_match):
        type_bonus = ENTITY_TYPE_BONUS
    elif type_match:
        type_bonus = ENTITY_TYPE_ONLY_BONUS
    else:
        type_bonus = 0.0

    return {
        "exact_text_match": exact_text_match,
        "partial_text_match": partial_text_match,
        "type_match": type_match,
        "text_bonus": float(text_bonus),
        "type_bonus": float(type_bonus),
        "bonus": float(text_bonus + type_bonus),
        "entity_type": entity_type,
    }


def infer_entity_type(
    entity_text: str,
    original_text: Optional[str],
    segmenter: Optional[str],
) -> Optional[str]:
    if not original_text:
        return None

    entity_norm = normalize_for_match(entity_text)

    if not entity_norm:
        return None

    prediction = predict_ner_entities(original_text, segmenter=segmenter)

    for entity in prediction.get("entities", []):
        candidate_values = [
            entity.get("text"),
            entity.get("segmented_text"),
        ]

        if any(normalize_for_match(value) == entity_norm for value in candidate_values):
            return entity.get("type")

    return None


def rerank_results_by_entity(
    results: List[Dict[str, Any]],
    entity_text: str,
    entity_type: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not results:
        return []

    base_scores = [float(result.get("score", 0.0)) for result in results]
    normalized_base_scores = normalize_result_scores(base_scores)
    reranked = []

    for result, base_score_norm in zip(results, normalized_base_scores):
        doc_idx = result.get("doc_index")

        if doc_idx is None:
            continue

        doc = state.metadata[int(doc_idx)]
        match_features = entity_match_features(doc, entity_text, entity_type)
        retrieval_score = float(result.get("score", 0.0))
        entity_bonus = float(match_features["bonus"])
        final_score = float(base_score_norm) + entity_bonus
        updated_result = dict(result)

        updated_result["retrieval_score"] = retrieval_score
        updated_result["base_score_norm"] = float(base_score_norm)
        updated_result["entity_bonus"] = entity_bonus
        updated_result["entity_match"] = match_features
        updated_result["rerank_method"] = "entity_boost"
        updated_result["score"] = final_score

        reranked.append(updated_result)

    return sorted(
        reranked,
        key=lambda item: (
            item["score"],
            item.get("entity_bonus", 0.0),
            item.get("retrieval_score", 0.0),
        ),
        reverse=True,
    )[:top_k]


def search_documents(
    query: str,
    method: str = "hybrid",
    top_k: int = 5,
    alpha: float = 0.6,
    segmenter: Optional[str] = None,
    candidate_k: Optional[int] = None,
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
            candidate_k=candidate_k if candidate_k is not None else max(50, top_k * 10),
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
        ensure_file_exists(CONFIG_PATH, "config.json")
        ensure_file_exists(NER_MODEL_DIR, "PhoBERT model directory")
        ensure_file_exists(EMBEDDING_MODEL_DIR, "Vietnamese embedding model directory")

        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            state.config = json.load(f)

        print(f"Loading FAISS index: {INDEX_PATH}")
        state.index = faiss.read_index(str(INDEX_PATH))

        print(f"Loading metadata: {METADATA_PATH}")
        state.metadata = load_jsonl(METADATA_PATH)

        print(f"Loading BM25: {BM25_PATH}")
        with BM25_PATH.open("rb") as f:
            state.bm25 = pickle.load(f)

        validate_retrieval_artifacts()

        print(f"Loading embedding model: {EMBEDDING_MODEL_DIR}")
        state.embedder = SentenceTransformer(str(EMBEDDING_MODEL_DIR), device=DEVICE)
        validate_embedding_model()

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
        "artifact_validation": state.artifact_validation,
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
        entity_type = normalize_text(payload.entity_type) if payload.entity_type else None

        if entity_type is None:
            entity_type = infer_entity_type(
                entity_text=payload.entity_text,
                original_text=payload.original_text,
                segmenter=selected_segmenter,
            )

        candidate_top_k = min(
            len(state.metadata),
            ENTITY_RERANK_MAX_CANDIDATES,
            max(payload.top_k * 10, ENTITY_RERANK_MIN_CANDIDATES),
        )

        candidate_results = search_documents(
            query=final_query,
            method=payload.method,
            top_k=candidate_top_k,
            alpha=payload.alpha,
            segmenter=selected_segmenter,
            candidate_k=candidate_top_k,
        )
        results = rerank_results_by_entity(
            results=candidate_results,
            entity_text=payload.entity_text,
            entity_type=entity_type,
            top_k=payload.top_k,
        )

        return {
            "entity_text": payload.entity_text,
            "entity_type": entity_type,
            "original_text": payload.original_text,
            "final_query": final_query,
            "segmenter": selected_segmenter,
            "segmented_query": word_segment_text(final_query, selected_segmenter),
            "method": payload.method,
            "top_k": payload.top_k,
            "candidate_top_k": candidate_top_k,
            "alpha": payload.alpha,
            "rerank_method": "entity_boost",
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
