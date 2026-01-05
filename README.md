# Recruitment Agent

RecruitmentAgent is a **LangChain-based recruitment assistant** that helps automate core hiring workflows: **extracting structured information** from resumes and job descriptions, **retrieving relevant evidence** using **RAG (Retrieval-Augmented Generation)**, **matching and ranking candidates by skills**, and **generating recruiter-ready email drafts**.

The project is organized as a clean, testable Python codebase with clear module boundaries (extraction → retrieval/RAG → evaluation → ranking → email generation), designed for iterative improvement of corpora, prompts, and scoring logic.
## Live Demo
Deployed App: [Recruitment Agent](https://recruitmentagent.streamlit.app/)

## Technology Highlights

- LangChain-based Retrieval-Augmented Generation (RAG)
- Pinecone vector index for scalable semantic search
- Embedding-based skill and document retrieval
- Modular Python architecture with extensive test coverage

---

## What this project does

### 1) Parse and normalize hiring inputs

* **Resume parsing** from PDFs and text sources.
* **Job description parsing** into structured fields.
* Normalization utilities to convert outputs into consistent “native” Python structures for downstream scoring and retrieval.

### 2) Retrieval-Augmented Generation (RAG)

* Builds and queries **domain corpora** (e.g., recruitment/tech corpus).
* Uses a vector store abstraction for **semantic retrieval**.
* Supports skill-focused retrieval via a dedicated **skill corpus**.

### 3) Skill matching and ranking

* Extracts and compares skills from CVs vs job requirements.
* Produces a ranked shortlist via a dedicated ranking module.
* Separates **matching logic** from **ranking policy**, making it easier to tune scoring without rewriting extraction or retrieval.

### 4) Evaluation layer (quality & reliability)

* Includes an **evaluator** component and a full **test suite** covering ingestion, retrieval behavior, vector store behavior, and skill queries.
* Designed to validate that changes to corpora, embeddings, chunking, or prompts do not regress results.

### 5) Recruiter workflow output

* Generates **email drafts** (e.g., outreach, interview invites) using extracted + retrieved context.

---

## Repository structure

```text
app/
  main.py

core/
  evaluator/
  email_generator.py
  ranker.py
  skill_matcher.py

  extractor/
    __init__.py
    cv_parser.py
    job_parser.py
    pdf_reader.py

  rag/
    __init__.py
    document_corpus.py
    retrieval.py
    skill_corpus.py
    vectorstore.py

  utils/
    helpers.py
    to_native.py
    __init__.py

data/
  rag_corpus/
    rag_corpus.csv
    rag_corpus.ipynb
  tech_corpus.csv
  resumes.csv
  tech_resumes1.csv
  eda.ipynb
  eda copy.ipynb

test/
  __init__.py
  test_ingest.py
  test_rag_evaluation.py
  test_rag_skills.py
  test_skills_query.py
  test_vectorstore.py

config.toml
requirements.txt
lib.txt
```

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/farahdimshawy/RecruitmentAgent.git
cd RecruitmentAgent
```

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Project configuration is managed via:

* `config.toml`

Common configuration includes model settings, embedding/vector store options, corpus paths, chunking parameters, and other runtime tuning knobs.

Update `config.toml` before running if you change any corpus files or retrieval settings.

---

## Running the project

The entry point is:

* `app/main.py`

Run:

```bash
python app/main.py
```

What happens at runtime depends on how `main.py` is wired (e.g., ingestion + retrieval demo, matching pipeline run, evaluation run). The pipeline modules are separated so you can extend `main.py` for your intended workflow (batch ranking, interactive matching, evaluation runs, etc.).

---

## Data and corpora

This repository includes multiple datasets and corpora used for ingestion, experimentation, and evaluation:

* `data/rag_corpus/rag_corpus.csv` – RAG document corpus
* `data/tech_corpus.csv` – skills/technology corpus
* `data/resumes.csv`, `data/tech_resumes1.csv` – resume datasets
* `data/*.ipynb` – EDA notebooks used to explore and curate datasets/corpora

If you update the corpus files, re-run ingestion logic (see tests and ingestion utilities) to ensure the vector store is aligned with the latest data.

---

## Engineering and programming skills demonstrated

### Python software engineering

* Modular package design with clear separation of concerns:

  * `extractor/` (input parsing)
  * `rag/` (retrieval + vector store)
  * `evaluator/` (quality checks)
  * matching/ranking/email generation components
* Utilities for normalization (`utils/to_native.py`) to ensure consistent internal data contracts.

### LangChain and RAG systems

* Practical RAG pipeline composition: corpus management → retrieval abstraction → skill-aware querying.
* Vector store encapsulation (`rag/vectorstore.py`) to isolate retrieval backend choices from business logic.

### Data processing and experimentation workflow

* Corpus and resume dataset handling with supporting EDA notebooks to iterate on quality.
* Structured corpora designed to support domain-specific retrieval rather than generic prompting.

### Testing and reliability

* Focus on test coverage for ingestion, retrieval behavior, and evaluation outcomes.
* Tests serve as guardrails for model/prompt/corpus iteration.

---

## Extending the project

Common extension points:

* Add new parsers (e.g., LinkedIn profile parsing) under `core/extractor/`
* Add new ranking policies in `core/ranker.py`
* Add new skill normalization logic in `core/skill_matcher.py`
* Add new retrieval strategies or backends in `core/rag/`
* Add evaluation metrics and regression checks in `core/evaluator/`

---
