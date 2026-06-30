# Vietnamese News Entity Retrieval

This project explores Vietnamese Named Entity Recognition (NER) for COVID-19 news and uses the extracted entities as metadata for more effective news retrieval. The retrieval demo combines BM25, FAISS vector search, and hybrid ranking.

The training and evaluation notebooks are in `notebooks/`. The runnable demo in `app/` uses a fine-tuned PhoBERT NER model, a Vietnamese sentence embedding model, BM25, and FAISS to extract entities from a user query and retrieve related news sentences.

## Dataset

The NER dataset is [VinAIResearch/PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19), a Vietnamese COVID-19 NER dataset with about 35K entities over 10K sentences and 10 entity types.

| Short name | Original / notebook label | Meaning |
| --- | --- | --- |
| PAT. | `PATIENT_ID` | Patient identifier |
| PER. | `PERSON_NAME` / `NAME` | Patient/contact person name |
| AGE | `AGE` | Age |
| GEN. | `GENDER` | Gender |
| OCC. | `OCCUPATION` / `JOB` | Occupation |
| LOC. | `LOCATION` | Location |
| ORG. | `ORGANIZATION` | Organization |
| SYM. | `SYMPTOM&DISEASE` / `SYMPTOM_AND_DISEASE` | Symptom or disease |
| TRA. | `TRANSPORTATION` | Transportation, flight code, vehicle code |
| DAT. | `DATE` | Date |

The notebooks load the HuggingFace mirror `phucdev/PhoNER_COVID19` with the `syllable` and `word` variants. The original dataset source remains the official VinAI repository. PhoNER_COVID19 is restricted to research and educational use; do not redistribute the original or modified dataset unless the dataset terms allow it.

## NER Models

The notebooks train and evaluate the following models.

| Group | Model | Short description |
| --- | --- | --- |
| Syllable-level | ELECTRA | Fine-tunes `NlpHUST/electra-base-vn` for token classification. |
| Syllable-level | XLM-RoBERTa | Fine-tunes `FacebookAI/xlm-roberta-base`. |
| Syllable-level | DistilBERT | Fine-tunes `distilbert/distilbert-base-multilingual-cased`. |
| Word-level | CNN | Multi-scale CNN1D with 3/5/7 kernels, residual connections, normalization, and token classification head. |
| Word-level | RNN | Improved BiRNN with embeddings, `pack_padded_sequence`, LayerNorm, and linear token classifier. |
| Word-level | LSTM | BiLSTM sequence labeler with attention/residual components and a classification head. |
| Word-level | GRU | BiGRU with a SwiGLU feed-forward block. |
| Word-level | PhoBERT | Fine-tunes `vinai/phobert-base-v2`; this is the NER checkpoint used by the API demo. |
| Word-level | ViDeBERTa | Fine-tunes `manhtt-079/vipubmed-deberta-base`. |

Word-level models use word-segmented inputs and align BIO labels to subword/token positions. Evaluation uses strict entity-level F1 on the test set.

## NER Benchmark

The table reports per-entity F1, Micro-F1, and Macro-F1 from the aggregate evaluation notebook.

### Syllable-level Models

| Model | PAT. | PER. | AGE | GEN. | OCC. | LOC. | ORG. | SYM. | TRA. | DAT. | Mic-F1 | Mac-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ELECTRA | 0.979 | 0.931 | **0.969** | **0.970** | **0.751** | **0.947** | 0.870 | **0.864** | 0.954 | **0.990** | **0.944** | **0.923** |
| XLM-RoBERTa | **0.980** | **0.952** | **0.969** | 0.966 | 0.731 | 0.945 | **0.872** | 0.853 | **0.969** | 0.988 | 0.943 | **0.923** |
| DistilBERT | 0.975 | 0.882 | 0.952 | 0.938 | 0.528 | 0.917 | 0.819 | 0.797 | 0.929 | 0.985 | 0.914 | 0.872 |

### Word-level Models

| Model | PAT. | PER. | AGE | GEN. | OCC. | LOC. | ORG. | SYM. | TRA. | DAT. | Mic-F1 | Mac-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.959 | 0.851 | 0.944 | 0.928 | 0.539 | 0.863 | 0.735 | 0.769 | 0.898 | 0.963 | 0.878 | 0.845 |
| RNN | 0.953 | 0.823 | 0.944 | 0.929 | 0.517 | 0.840 | 0.723 | 0.754 | 0.866 | 0.951 | 0.864 | 0.830 |
| LSTM | 0.958 | 0.810 | 0.946 | 0.938 | 0.510 | 0.869 | 0.760 | 0.777 | 0.834 | 0.961 | 0.882 | 0.836 |
| GRU | 0.952 | 0.827 | 0.941 | 0.936 | 0.490 | 0.853 | 0.714 | 0.754 | 0.850 | 0.953 | 0.868 | 0.827 |
| PhoBERT | **0.983** | 0.904 | **0.969** | **0.972** | **0.736** | **0.952** | **0.885** | **0.883** | **0.979** | **0.987** | **0.949** | **0.925** |
| ViDeBERTa | 0.982 | **0.911** | 0.964 | 0.966 | 0.724 | 0.935 | 0.865 | 0.863 | 0.964 | 0.986 | 0.938 | 0.916 |

PhoBERT obtains the best Macro-F1 among the word-level models and is the checkpoint used by the retrieval demo. The transformer models are generally more stable than the custom sequence models. Low-resource labels such as `OCCUPATION` / `JOB` remain the weakest entity category.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `Mini Project NLP [Evaluate All Models].ipynb` | Loads checkpoints, standardizes evaluation, and aggregates benchmark results. |
| `Mini Project NLP [PhoBERT].ipynb` | Fine-tunes PhoBERT for word-level NER. |
| `Mini Project NLP [ViDeBERTa].ipynb` | Fine-tunes ViDeBERTa for word-level NER. |
| `Mini Project NLP [ELECTRA].ipynb` | Fine-tunes ELECTRA for syllable-level NER. |
| `Mini Project NLP [XLM-RoBERTa].ipynb` | Fine-tunes XLM-RoBERTa for syllable-level NER. |
| `Mini Project NLP [DistilBERT].ipynb` | Fine-tunes multilingual DistilBERT for syllable-level NER. |
| `Mini Project NLP [CNN].ipynb` | Trains the multi-scale CNN1D NER model. |
| `Mini Project NLP [RNN].ipynb` | Trains the BiRNN NER model. |
| `Mini Project NLP [LSTM].ipynb` | Trains the BiLSTM NER model. |
| `Mini Project NLP [GRU].ipynb` | Trains the BiGRU + SwiGLU NER model. |
| `News_Retrieval.ipynb` | Builds the retrieval corpus, entity metadata, BM25 index, FAISS index, and notebook demo. |

## News Retrieval Demo

The demo turns the `News_Retrieval.ipynb` workflow into a FastAPI backend and Streamlit UI.

1. Receive a Vietnamese user query.
2. Word-segment the query with `underthesea`, `pyvi`, or `vncorenlp`.
3. Run PhoBERT NER to extract query entities.
4. Search either the full query or a selected entity.
5. Retrieve documents with `bm25`, `vector`, or `hybrid`.
6. Return news text, score details, segmented text, and entity metadata.

Runtime retrieval artifacts:

```text
retrieval_artifacts/
├── bm25.pkl
├── config.json
├── metadata.jsonl
└── news.index
```

Artifact construction in the notebook:

- The demo corpus is built by merging the `train`, `validation`, and `test` splits of PhoNER_COVID19.
- `metadata.jsonl` stores `display_text`, `segmented_text`, entity spans, entity texts, segmented entity texts, and entity types.
- BM25 uses `rank_bm25.BM25Okapi` over word-segmented tokens.
- Vector search uses `dangvantuan/vietnamese-embedding`, normalized sentence embeddings, and FAISS `IndexFlatIP`.
- Hybrid search uses weighted Reciprocal Rank Fusion (RRF) over BM25 and FAISS candidate rankings. `alpha` controls the vector/BM25 weight.
- Entity search retrieves an expanded candidate set, then reranks it with entity text/type boosts.
- API startup validates artifact consistency: metadata rows, FAISS size/dimension, BM25 corpus size, config values, and embedding dimension.

## Project Layout

```text
.
├── app/
│   ├── api.py
│   └── ui.py
├── notebooks/
│   ├── Mini Project NLP [*.ipynb]
│   └── News_Retrieval.ipynb
├── scripts/
│   └── download_models.py
├── retrieval_artifacts/
│   ├── bm25.pkl
│   ├── config.json
│   ├── metadata.jsonl
│   └── news.index
├── models/
│   ├── PhoBERT/
│   ├── Vietnamese_Embedding/
│   └── VnCoreNLP/
├── Makefile
├── requirements-api.txt
├── requirements-ui.txt
├── requirements-download.txt
└── README.md
```

`retrieval_artifacts/` and `models/` are runtime assets and may be absent after cloning the repository.

## Setup And Run

Install API dependencies:

```bash
make install-api
```

Install UI dependencies:

```bash
make install-ui
```

Or install manually:

```bash
pip install -r requirements-api.txt
pip install -r requirements-ui.txt
```

Download runtime models and retrieval artifacts:

```bash
make download-models
```

The download script fetches:

- Retrieval artifacts into `retrieval_artifacts/`
- PhoBERT tokenizer, `config.json`, and `model.safetensors` into `models/PhoBERT/`
- Vietnamese embedding model into `models/Vietnamese_Embedding/`
- VnCoreNLP jar and wordsegmenter files into `models/VnCoreNLP/`

Run the API:

```bash
make run-api
```

Device-specific API commands:

```bash
make run-api-cpu
make run-api-gpu
make run-api-mps
```

Run the UI in another terminal:

```bash
make run-ui
```

Defaults:

```text
API: http://127.0.0.1:8000
UI:  http://127.0.0.1:8501
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Path debugging:

```text
http://127.0.0.1:8000/debug/paths
```

## VnCoreNLP And Java

`underthesea` and `pyvi` do not require Java. `vncorenlp` requires Java 1.8+ and local files in `models/VnCoreNLP/`.

On macOS, OpenJDK 17 is a practical default:

```bash
brew install openjdk@17
```

Environment setup:

```bash
echo 'export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"' >> ~/.zshrc
echo 'export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

The API initializes VnCoreNLP once at startup to avoid JVM/pyjnius race conditions. If VnCoreNLP initialization fails, the API still runs and `underthesea` / `pyvi` remain available. The `vncorenlp` segmenter is then reported as unavailable in `/health`.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Returns model, artifact, device, segmenter, and artifact validation status. |
| `GET /debug/paths` | Shows resolved paths and relevant environment variables. |
| `POST /segment` | Word-segments an input sentence. |
| `POST /ner` | Runs PhoBERT NER. |
| `POST /search` | Searches the full query with `bm25`, `vector`, or `hybrid`. |
| `POST /entity-search` | Searches with a selected entity and applies entity-aware reranking. |

Example `entity-search` payload:

```json
{
  "entity_text": "Ha Noi",
  "entity_type": "LOCATION",
  "original_text": "Benh nhan 129 o Ha Noi tung nhap canh qua san bay Noi Bai.",
  "method": "hybrid",
  "top_k": 5,
  "alpha": 0.6,
  "segmenter": "underthesea"
}
```

## Runtime Configuration

You can override runtime paths and behavior with environment variables or Makefile arguments.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARTIFACT_DIR` | `retrieval_artifacts` | Directory containing BM25, FAISS index, config, and metadata. |
| `MODEL_DIR` | `models` | Runtime model directory. |
| `NER_MODEL_DIR` | `models/PhoBERT` | Fine-tuned PhoBERT NER model. |
| `EMBEDDING_MODEL_DIR` | `models/Vietnamese_Embedding` | SentenceTransformer embedding model. |
| `VNCORENLP_DIR` | `models/VnCoreNLP` | Local VnCoreNLP files. |
| `DEFAULT_SEGMENTER` | `underthesea` | Default word segmenter. |
| `DEVICE` | `cpu` | `cpu`, `mps`, `cuda`, `gpu`, or `auto`. |
| `RRF_K` | `60` | Reciprocal Rank Fusion smoothing constant. |
| `ENTITY_RERANK_MIN_CANDIDATES` | `50` | Minimum candidate pool size for entity search. |
| `ENTITY_RERANK_MAX_CANDIDATES` | `200` | Maximum candidate pool size for entity search. |
| `ENTITY_EXACT_TEXT_BONUS` | `0.35` | Score bonus for exact entity text match. |
| `ENTITY_PARTIAL_TEXT_BONUS` | `0.15` | Score bonus for partial entity text match. |
| `ENTITY_TYPE_BONUS` | `0.10` | Score bonus when text match also has the same entity type. |
| `ENTITY_TYPE_ONLY_BONUS` | `0.03` | Small fallback bonus for same entity type without text match. |

Examples:

```bash
make run-api API_PORT=9000 DEVICE=auto DEFAULT_SEGMENTER=pyvi
make run-ui API_PORT=9000
```

## Remaining Retrieval Improvements

The current demo is stronger than a plain BM25/vector baseline, but further work is still useful:

- Build a dedicated retrieval evaluation set instead of evaluating retrieval over a merged train/validation/test corpus.
- Add structured field indexing for `date`, `patient_id`, `transportation`, and normalized entity aliases.
- Tune `alpha`, RRF parameters, candidate pool size, and entity bonus weights on a validation set with relevance labels.
- Normalize dates, patient IDs, flight codes, casing, Vietnamese diacritics, and spelling variants more aggressively.
- Add a Vietnamese cross-encoder or reranker over the top 20-50 candidates if higher latency is acceptable.

## Data Note

The demo artifact may contain original or word-segmented sentences derived from PhoNER_COVID19. Before publishing artifacts or deploying the demo for external users, verify the dataset terms and avoid redistributing dataset-derived content without the appropriate permission.
