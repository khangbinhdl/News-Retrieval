import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import gdown
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]

PHOBERT_TOKENIZER_NAME = os.getenv("PHOBERT_TOKENIZER_NAME", "vinai/phobert-base-v2")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "dangvantuan/vietnamese-embedding")

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))

NER_MODEL_DIR = Path(os.getenv("NER_MODEL_DIR", MODEL_DIR / "PhoBERT"))
EMBEDDING_MODEL_DIR = Path(os.getenv("EMBEDDING_MODEL_DIR", MODEL_DIR / "Vietnamese_Embedding"))
VNCORENLP_DIR = Path(os.getenv("VNCORENLP_DIR", MODEL_DIR / "VnCoreNLP"))

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT_DIR / "retrieval_artifacts"))
DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", ROOT_DIR / "downloads"))

PHOBERT_MODEL_FILES = {
    "model.safetensors": "10qADwgwRf5GUzww5oBAUmlWl_zq6B_fN",
    "config.json": "1Otx84Seiff2S_oh4cIJ90iVRSc6QJDmb",
}

RETRIEVAL_ARTIFACT_FILE_ID = os.getenv(
    "RETRIEVAL_ARTIFACT_FILE_ID",
    "1KhgExefN5v6u_MyL407t8bKWHW5mJNX7",
)
RETRIEVAL_ARTIFACT_ZIP_NAME = os.getenv(
    "RETRIEVAL_ARTIFACT_ZIP_NAME",
    "retrieval_artifact.zip",
)

VNCORENLP_FILES = {
    "VnCoreNLP-1.2.jar": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr",
}

REQUIRED_RETRIEVAL_ARTIFACT_FILES = [
    "bm25.pkl",
    "config.json",
    "metadata.jsonl",
    "news.index",
]


def has_any_file(directory: Path, filenames: list[str]) -> bool:
    return any((directory / filename).exists() for filename in filenames)


def file_exists_and_non_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def phobert_tokenizer_exists() -> bool:
    tokenizer_model_files = [
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "vocab.txt",
        "bpe.codes",
    ]

    tokenizer_config_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    return (
        NER_MODEL_DIR.exists()
        and has_any_file(NER_MODEL_DIR, tokenizer_model_files)
        and all((NER_MODEL_DIR / filename).exists() for filename in tokenizer_config_files)
    )


def phobert_weights_exist() -> bool:
    return all(
        file_exists_and_non_empty(NER_MODEL_DIR / filename)
        for filename in PHOBERT_MODEL_FILES
    )


def embedding_model_exists() -> bool:
    model_files = [
        "model.safetensors",
        "pytorch_model.bin",
    ]

    return (
        EMBEDDING_MODEL_DIR.exists()
        and (EMBEDDING_MODEL_DIR / "modules.json").exists()
        and has_any_file(EMBEDDING_MODEL_DIR, model_files)
    )


def vncorenlp_exists() -> bool:
    return all((VNCORENLP_DIR / rel_path).exists() for rel_path in VNCORENLP_FILES)


def retrieval_artifacts_exist() -> bool:
    return all(
        file_exists_and_non_empty(ARTIFACT_DIR / filename)
        for filename in REQUIRED_RETRIEVAL_ARTIFACT_FILES
    )


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_exists_and_non_empty(output_path):
        print(f"Exists. Skip: {output_path}")
        return

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading Google Drive file id={file_id}")
    print(f"To: {output_path}")

    result = gdown.download(
        url=url,
        output=str(output_path),
        quiet=False,
    )

    if result is None:
        raise RuntimeError(f"gdown failed to download file: {output_path}")

    if not file_exists_and_non_empty(output_path):
        raise RuntimeError(f"Download failed or empty file: {output_path}")


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = output_dir / member.filename
            resolved_member_path = member_path.resolve()
            resolved_output_dir = output_dir.resolve()

            if not str(resolved_member_path).startswith(str(resolved_output_dir)):
                raise RuntimeError(f"Unsafe zip path detected: {member.filename}")

        zip_ref.extractall(output_dir)


def flatten_artifact_dir_if_needed() -> None:
    """
    Handle both zip structures:

    Case 1:
        retrieval_artifact.zip
        ├── bm25.pkl
        ├── config.json
        ├── metadata.jsonl
        └── news.index

    Case 2:
        retrieval_artifact.zip
        └── retrieval_artifacts/
            ├── bm25.pkl
            ├── config.json
            ├── metadata.jsonl
            └── news.index

    Case 3:
        retrieval_artifact.zip
        └── news_faiss_demo/
            ├── bm25.pkl
            ├── config.json
            ├── metadata.jsonl
            └── news.index
    """
    if retrieval_artifacts_exist():
        return

    candidate_dirs = [
        ARTIFACT_DIR / "retrieval_artifacts",
        ARTIFACT_DIR / "retrieval_artifact",
        ARTIFACT_DIR / "news_faiss_demo",
        ARTIFACT_DIR / "artifact",
        ARTIFACT_DIR / "artifacts",
    ]

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            continue

        if all(file_exists_and_non_empty(candidate_dir / filename) for filename in REQUIRED_RETRIEVAL_ARTIFACT_FILES):
            print(f"Flattening retrieval artifacts from: {candidate_dir}")

            for filename in REQUIRED_RETRIEVAL_ARTIFACT_FILES:
                source = candidate_dir / filename
                target = ARTIFACT_DIR / filename

                if not target.exists():
                    source.replace(target)

            return


def download_retrieval_artifacts() -> None:
    if retrieval_artifacts_exist():
        print(f"Retrieval artifacts already exist. Skip: {ARTIFACT_DIR}")
        return

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOADS_DIR / RETRIEVAL_ARTIFACT_ZIP_NAME

    download_google_drive_file(
        file_id=RETRIEVAL_ARTIFACT_FILE_ID,
        output_path=zip_path,
    )

    print(f"Extracting retrieval artifacts to: {ARTIFACT_DIR}")
    safe_extract_zip(zip_path, ARTIFACT_DIR)
    flatten_artifact_dir_if_needed()

    if not retrieval_artifacts_exist():
        found_files = sorted(str(path.relative_to(ARTIFACT_DIR)) for path in ARTIFACT_DIR.rglob("*") if path.is_file())
        raise RuntimeError(
            "Retrieval artifacts are still incomplete after extraction.\n"
            f"Expected files: {REQUIRED_RETRIEVAL_ARTIFACT_FILES}\n"
            f"Artifact dir: {ARTIFACT_DIR}\n"
            f"Found files: {found_files}"
        )

    print("Retrieval artifacts downloaded and extracted.")


def download_phobert_tokenizer() -> None:
    if phobert_tokenizer_exists():
        print(f"PhoBERT tokenizer already exists. Skip: {NER_MODEL_DIR}")
        return

    NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PhoBERT tokenizer: {PHOBERT_TOKENIZER_NAME}")
    print(f"Saving tokenizer files to: {NER_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_TOKENIZER_NAME, use_fast=False)
    tokenizer.save_pretrained(str(NER_MODEL_DIR))

    print("PhoBERT tokenizer downloaded.")


def download_phobert_weights() -> None:
    if phobert_weights_exist():
        print(f"PhoBERT config and weights already exist. Skip: {NER_MODEL_DIR}")
        return

    NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename, file_id in PHOBERT_MODEL_FILES.items():
        download_google_drive_file(
            file_id=file_id,
            output_path=NER_MODEL_DIR / filename,
        )

    print("PhoBERT config and weights downloaded.")


def download_embedding_model() -> None:
    if embedding_model_exists():
        print(f"Vietnamese embedding model already exists. Skip: {EMBEDDING_MODEL_DIR}")
        return

    EMBEDDING_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Saving embedding model to: {EMBEDDING_MODEL_DIR}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.save(str(EMBEDDING_MODEL_DIR))

    print("Vietnamese embedding model downloaded.")


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_exists_and_non_empty(output_path):
        print(f"Exists. Skip: {output_path}")
        return

    print(f"Downloading: {url}")
    print(f"To: {output_path}")

    urlretrieve(url, output_path)

    if not file_exists_and_non_empty(output_path):
        raise RuntimeError(f"Download failed or empty file: {output_path}")


def download_vncorenlp() -> None:
    if vncorenlp_exists():
        print(f"VnCoreNLP already exists. Skip: {VNCORENLP_DIR}")
        return

    print(f"Downloading VnCoreNLP to: {VNCORENLP_DIR}")

    for rel_path, url in VNCORENLP_FILES.items():
        download_file(url, VNCORENLP_DIR / rel_path)

    print("VnCoreNLP downloaded.")


def print_summary() -> None:
    print("\nDownload summary")
    print("-" * 80)
    print(f"Retrieval artifact directory: {ARTIFACT_DIR}")
    print(f"Retrieval artifacts exist: {retrieval_artifacts_exist()}")

    for filename in REQUIRED_RETRIEVAL_ARTIFACT_FILES:
        path = ARTIFACT_DIR / filename
        print(f"  - {filename}: exists={path.exists()} size={path.stat().st_size if path.exists() else None}")

    print(f"PhoBERT directory: {NER_MODEL_DIR}")
    print(f"PhoBERT tokenizer exists: {phobert_tokenizer_exists()}")
    print(f"PhoBERT weights exist: {phobert_weights_exist()}")
    print(f"Vietnamese embedding directory: {EMBEDDING_MODEL_DIR}")
    print(f"Vietnamese embedding exists: {embedding_model_exists()}")
    print(f"VnCoreNLP directory: {VNCORENLP_DIR}")
    print(f"VnCoreNLP exists: {vncorenlp_exists()}")
    print("-" * 80)


def main() -> None:
    download_retrieval_artifacts()
    download_phobert_tokenizer()
    download_phobert_weights()
    download_embedding_model()
    download_vncorenlp()
    print_summary()
    print("Done.")


if __name__ == "__main__":
    main()