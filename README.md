# News Entity Retrieval Demo

A local demo for Vietnamese news retrieval using:

- PhoBERT NER model
- Vietnamese sentence embedding model
- FAISS vector search
- BM25 lexical search
- Hybrid retrieval
- Selectable Vietnamese word segmentation:
  - `underthesea`
  - `pyvi`
  - `vncorenlp`

## Project layout

Expected layout:

```text
.
├── retrieval_artifacts/
│   ├── bm25.pkl
│   ├── config.json
│   ├── metadata.jsonl
│   └── news.index
├── models/
│   ├── PhoBERT/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── sentencepiece.bpe.model
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   └── other tokenizer files
│   ├── Vietnamese_Embedding/
│   │   └── embedding model downloaded by scripts/download_models.py
│   └── VnCoreNLP/
│       ├── VnCoreNLP-1.2.jar
│       └── models/
│           └── wordsegmenter/
│               ├── vi-vocab
│               └── wordsegmenter.rdr
├── notebooks/
│   └── News_Retrieval.ipynb
├── app/
│   ├── api.py
│   └── ui.py
├── scripts/
│   └── download_models.py
├── requirements-api.txt
├── requirements-ui.txt
├── requirements-download.txt
├── Makefile
└── README.md
```

## 1. Install dependencies

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

If you only want to run the model/artifact downloader in a separate environment, install:

```bash
pip install -r requirements-download.txt
```

## 2. Java setup for VnCoreNLP on macOS

VnCoreNLP requires Java 1.8+.

For macOS, OpenJDK 17 is recommended:

```bash
brew install openjdk@17
```

Create a symlink so macOS can detect the JDK:

```bash
sudo ln -sfn "$(brew --prefix openjdk@17)/libexec/openjdk.jdk" /Library/Java/JavaVirtualMachines/openjdk-17.jdk
```

Set Java environment variables:

```bash
echo 'export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"' >> ~/.zshrc
echo 'export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Check Java:

```bash
java -version
echo $JAVA_HOME
echo $JVM_PATH
```

You should see Java 17.

If you previously installed OpenJDK 26 and created a symlink, remove the old symlink:

```bash
sudo rm -f /Library/Java/JavaVirtualMachines/openjdk.jdk
sudo rm -f /Library/Java/JavaVirtualMachines/openjdk-26.jdk
```

If you want to uninstall OpenJDK 26:

```bash
brew uninstall openjdk
brew autoremove
```

## 3. Download local models and retrieval artifacts

Run once before starting the API:

```bash
make download-models
```

This downloads:

- Retrieval artifacts into `retrieval_artifacts/`
- PhoBERT tokenizer files into `models/PhoBERT/`
- PhoBERT `config.json` and `model.safetensors` into `models/PhoBERT/` using Google Drive via `gdown`
- Vietnamese embedding model into `models/Vietnamese_Embedding/`
- VnCoreNLP files into `models/VnCoreNLP/`

Required retrieval artifact files:

```text
retrieval_artifacts/bm25.pkl
retrieval_artifacts/config.json
retrieval_artifacts/metadata.jsonl
retrieval_artifacts/news.index
```

Google Drive file IDs:

```text
retrieval_artifact.zip: 1KhgExefN5v6u_MyL407t8bKWHW5mJNX7
model.safetensors: 10qADwgwRf5GUzww5oBAUmlWl_zq6B_fN
config.json: 1Otx84Seiff2S_oh4cIJ90iVRSc6QJDmb
```

If you already have retrieval artifacts locally, place them into:

```text
retrieval_artifacts/
```

Or run the API with a custom artifact folder:

```bash
make run-api ARTIFACT_DIR=your_artifact_folder
```

## 4. Run API

Default CPU mode:

```bash
make run-api
```

Explicit CPU mode:

```bash
make run-api-cpu
```

Auto GPU mode:

```bash
make run-api-gpu
```

Apple Silicon MPS mode:

```bash
make run-api-mps
```

You can customize host and port:

```bash
make run-api API_HOST=127.0.0.1 API_PORT=9000
```

If you changed the artifact or model folders:

```bash
make run-api ARTIFACT_DIR=retrieval_artifacts MODEL_DIR=models
```

## 5. Run UI

Open another terminal:

```bash
make run-ui
```

Default API URL used by Streamlit:

```text
http://127.0.0.1:8000
```

If you changed the API port:

```bash
make run-ui API_PORT=9000
```

## 6. Word segmentation options

The UI allows selecting one of:

```text
underthesea
pyvi
vncorenlp
```

Notes:

- `underthesea` does not require Java.
- `pyvi` does not require Java.
- `vncorenlp` requires Java and local files in `models/VnCoreNLP/`.
- VnCoreNLP is initialized lazily. The API starts without loading Java; it loads VnCoreNLP only when users select `vncorenlp`.

## 7. Health check

After starting the API, open:

```text
http://127.0.0.1:8000/health
```

Or run:

```bash
curl http://127.0.0.1:8000/health
```

The response shows:

- current device
- number of documents
- FAISS index size
- available segmenters
- model folders
- VnCoreNLP file status

For path debugging, open:

```text
http://127.0.0.1:8000/debug/paths
```