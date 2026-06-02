# UK Public Procurement RAG System - BiP Solutions Tech Assignment

A Retrieval-Augmented Generation (RAG) system for querying UK Find a Tender Service (FTS) procurement contracts using natural language. The solution processes Open Contracting Data Standard (OCDS) procurement notices, combines hybrid retrieval techniques (BM25 + semantic search + cross-encoder reranking), and generates grounded answers using Mistral-7B.

# Problem Statement

Public procurement data is large, semi-structured, and difficult to search using traditional keyword-based approaches. Users often ask questions such as:

* Find framework agreements for IT services in the NHS
* Contracts awarded in London for construction
* Multi-supplier frameworks for cloud services

These queries contain both semantic intent and structured procurement concepts. The objective of this project was to build a Retrieval-Augmented Generation (RAG) system capable of understanding natural language procurement queries and returning relevant procurement notices with supporting explanations.

# Dataset

The system uses UK Find a Tender Service (FTS) procurement notices published in Open Contracting Data Standard (OCDS) format.

### Dataset Characteristics

| Metric            | Value                      |
| ----------------- | -------------------------- |
| Source            | UK Find a Tender Service   |
| Format            | JSONL (OCDS)               |
| Records Processed | 35,839                     |
| Data Type         | Public Procurement Notices |
| Structure         | Nested JSON                |

The raw OCDS records were flattened into a contract-centric table containing:

* Notice ID
* Title
* Description
* Buyer
* Suppliers
* CPV Classifications
* Procurement Method
* Region
* Contract Values
* Additional metadata


# System Architecture

```text
User Query
     │
     ▼
Intent Parsing
     │
     ▼
Hybrid Retrieval
 ┌──────────────┐
 │ BM25 Search  │
 └──────────────┘
        +
 ┌──────────────┐
 │ BGE Embedding│
 │  Search      │
 └──────────────┘
     │
     ▼
Reciprocal Rank Fusion
     │
     ▼
Cross Encoder Re-ranking
     │
     ▼
Top-k Contracts
     │
     ▼
Mistral 7B
     │
     ▼
Grounded Procurement Answer
```

The retrieval layer and generation layer were deliberately separated. Retrieval is responsible for identifying relevant procurement notices, while generation is responsible for synthesising a grounded answer using only retrieved evidence. This separation improves transparency and reduces hallucination risk.


# Architecture Decision Log

## Data Representation

The original OCDS procurement notices contain deeply nested structures with varying schemas across notices. To simplify retrieval and downstream processing, the raw JSON records were flattened into a contract-centric table where each row represents a procurement notice and key procurement attributes are stored as structured columns.

## Hybrid Retrieval

A hybrid retrieval architecture was selected instead of relying solely on either keyword search or vector search. Procurement queries frequently contain exact terminology such as supplier names, framework identifiers, and CPV codes, which BM25 handles effectively. Semantic embeddings complement BM25 by supporting concept-level matching and natural language understanding.

## Embedding Model Selection

BAAI/bge-small-en-v1.5 was selected for dense retrieval. The model provides stronger retrieval performance than older MiniLM-based models while remaining lightweight enough for experimentation and local deployment.

## Vector Database Selection

ChromaDB was selected as the vector store because it supports persistent storage, metadata-aware retrieval workflows, and straightforward Python integration. This provides a more realistic retrieval architecture than a simple in-memory FAISS index.

## Ranking Strategy

Initial candidates are retrieved independently using BM25 and dense embeddings. Reciprocal Rank Fusion (RRF) is used to combine rankings from both retrieval methods. A cross-encoder model is then applied to rerank the fused candidate set, improving precision within the final results.

## Generation Model Selection

Mistral-7B-Instruct-v0.3 was selected for answer generation. The model synthesises grounded answers from retrieved procurement notices and provides concise explanations of why contracts were matched.

## Evaluation Methodology

No labelled benchmark exists for this procurement dataset. A manual relevance assessment approach was therefore adopted. Representative procurement queries were selected, retrieved results were manually labelled, and retrieval quality was measured using Precision@5, Mean Reciprocal Rank (MRR), and NDCG@5.


# Project Structure

```text
uk-procurement-rag/

├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Rithik_Sah_Bip_Solutions_Data_Science_Tech_Assignment.ipynb
│
├── src/
│   ├── ingest.py
│   ├── preprocessing.py
│   ├── intent.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   └── evaluation.py
```


# End-to-End Workflow

The full implementation, experimentation process, retrieval pipeline, evaluation workflow, and example outputs are available in:

```text
notebooks/Bip_Solutions_Data_Science_Tech_Assignment.ipynb
```

Reviewers are encouraged to refer to the notebook for the complete end-to-end implementation.


# Setup & Usage

## 1. Clone Repository

```bash
git clone <repository-url>
cd uk-procurement-rag
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Create Hugging Face API Token

This project uses:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

for answer generation.

Create a Hugging Face account and generate an access token:

https://huggingface.co/settings/tokens

Export the token:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

or

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"
```

## 4. Run Notebook

Open:

```text
notebooks/Rithik_Sah_Bip_Solutions_Data_Science_Tech_Assignment.ipynb
```

and execute cells sequentially.

The notebook contains the complete workflow:

1. Data ingestion
2. Data flattening
3. Data cleaning
4. Embedding generation
5. ChromaDB indexing
6. BM25 retrieval
7. Hybrid retrieval
8. Cross-encoder reranking
9. Mistral answer generation
10. Evaluation


# Evaluation Results

## Per Query Results

| Query                                                | Precision@5 (Relaxed) | Precision@5 (Strict) | MRR   | NDCG@5 |
| ---------------------------------------------------- | --------------------- | -------------------- | ----- | ------ |
| Contracts awarded in London for construction         | 1.000                 | 0.600                | 0.333 | 0.774  |
| Find framework agreements for IT services in the NHS | 0.400                 | 0.400                | 1.000 | 0.918  |
| Multi-supplier frameworks for cloud services         | 0.600                 | 0.000                | 0.143 | 0.947  |

## Overall Results

| Metric                | Score |
| --------------------- | ----- |
| Precision@5 (Relaxed) | 0.667 |
| Precision@5 (Strict)  | 0.333 |
| MRR                   | 0.492 |
| NDCG@5                | 0.880 |

The results indicate strong ranking quality overall while highlighting challenges associated with domain-specific terminology and sparse procurement metadata.


# Known Failure Modes

The system can struggle when procurement terminology differs significantly from user terminology. For example, cloud-related procurements are frequently described as digital services, platform services, hosting services, or software solutions rather than explicitly using the term cloud. Missing CPV codes, incomplete supplier information, and sparse regional metadata can also reduce retrieval precision.


# Future Improvements

Given additional time, the following enhancements would be prioritised:

* Metadata-aware filtering during retrieval
* Procurement ontology expansion
* Query rewriting using LLMs
* Larger manually labelled evaluation datasets
* Domain-specific embedding fine-tuning
* Streamlit-based user interface
* Automated evaluation pipeline
* Containerised deployment
