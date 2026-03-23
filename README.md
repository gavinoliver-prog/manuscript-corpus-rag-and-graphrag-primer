# Scientific RAG from PDFs: A Transparent, Package-Native Walkthrough

This repository contains a notebook demonstrating a **Graph-Augmented Retrieval-Augmented Generation (GraphRAG)** pipeline applied to a corpus of scientific papers in precision oncology and rare disease genomics.

It is designed as an educational resource. Every data transformation is made visible, every package choice is explained, and prompt engineering decisions are treated as first-class tutorial content.

## The Problem

Scientific literature in precision oncology and rare disease genomics grows faster than any researcher can read. Key findings about fusion transcript detection, gene–disease associations, and computational methods are scattered across dozens of papers. This pipeline offers a principled way to:

- **Search by meaning** (not just keywords) across a personal publication corpus
- **Ground LLM answers** in retrieved evidence, reducing hallucination
- **Build a knowledge graph** from the full corpus and use it to expand retrieval beyond what vector similarity alone finds

## Repository Structure

```
manuscript-corpus-rag-and-graphrag-primer/
│
├── data/
│   └── papers/                     # input PDFs
│
├── outputs/                        # written by the notebook at runtime
│   ├── chunks.csv
│   ├── chunk_embeddings.npy
│   ├── triplets_raw.csv
│   ├── triplets_filtered.csv
│   ├── triplets_clean.csv
│   ├── graph.graphml
│   └── top_hits.csv
│
├── scientific_rag_notebook.ipynb
├── requirements.txt
└── README.md
```

Intermediate artifacts are cached to `outputs/` so expensive steps (embedding, triplet extraction, normalisation) are skipped on re-runs.

## Environment Setup

This project uses **uv** for environment management.

### Install Python if needed

```bash
uv python install 3.11
```

Python 3.11 is recommended. Python 3.14 works but may produce deprecation warnings from the spaCy/Pydantic v1 stack.

### Create the virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

### Configure your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

The notebook loads this with `python-dotenv`. It is `.gitignore`d by default.

### Launch the notebook

```bash
uv run jupyter lab
```

Then open `scientific_rag_notebook.ipynb` and run cells top to bottom.

## Pipeline Overview

The notebook runs in two phases:

**Phase 1 — Build the indices (run once, results cached)**

```
PDF corpus
  → Text Extraction            (PyMuPDF4LLM)
  → Document Wrapping          (LlamaIndex Document)
  → Chunking                   (SentenceSplitter, 800 tokens / 100 overlap)
  → Embeddings                 (all-MiniLM-L6-v2)          ← vector index
  → Full-corpus triplet extraction   (GPT-4o-mini)
  → Junk filter                (rule-based)
  → Entity normalisation       (GPT-4o-mini, batched)
  → Predicate normalisation    (GPT-4o-mini, canonical vocab)
  → Knowledge Graph            (NetworkX MultiDiGraph)      ← graph index
```

**Phase 2 — Query (run per question)**

```
User question
  → Query Rewriting            (GPT-4o-mini)     ← Prompt Engineering §1
  → Vector Retrieval           (cosine similarity)
  → Context Assembly                              ← Prompt Engineering §2
  → Grounded Answer            (GPT-4.1)          ← Prompt Engineering §3
  → Graph-Augmented Retrieval  (entity expansion via G)
  → Enriched Answer            (GPT-4.1)
  → Side-by-side evaluation
```

## Notebook Sections

| § | Title | Key output |
|---|-------|-----------|
| 1 | Environment Setup | `client`, imports |
| 2 | Discover the PDF Corpus | `corpus_df` |
| 3 | Extract Text with PyMuPDF4LLM | `docs_df` |
| 4 | Wrap into LlamaIndex Documents | `documents` |
| 5 | Chunking with SentenceSplitter | `chunks_df` |
| 6 | Embeddings with Sentence Transformers | `chunk_embeddings` |
| 7 | Prompt Engineering Step 1: Query Rewriting | `rewritten_query` |
| 8 | Semantic Retrieval | `top_hits` |
| 9 | Prompt Engineering Step 2: Context Assembly | `context` |
| 10 | Prompt Engineering Step 3: Grounded Answer Generation | `answer` |
| 11 | Full-Corpus Triplet Extraction | `triplets_raw_df` |
| 12 | Junk Filter | `triplets_filtered` |
| 13 | Entity and Predicate Normalisation | `triplets_clean`, `entity_map`, `predicate_map` |
| 14 | Building the Knowledge Graph | `G` (MultiDiGraph) |
| 15 | Exploring the Knowledge Graph | neighborhood queries, shortest path |
| 16 | Graph-Augmented Retrieval | `graph_hits`, `expansion_entities` |
| 17 | Vector-Only vs Graph-Augmented Answer | side-by-side comparison |
| 18 | Evaluation | programmatic + LLM-as-judge metrics |
| 19 | Save Outputs | all artifacts written to `outputs/` |
| 20 | Just Ask Questions | end-to-end `ask()` function |

Each section follows the same pattern:

> **Input →** what object goes in
> **Method →** which package/technique is applied, and why
> **Output →** what object comes out

## Key Design Decisions

**Batched entity normalisation (§13).** With 2000+ unique entities, a single LLM call would exceed the output token limit and return truncated JSON. Entities are processed in batches of 200, results merged in-memory.

**Predicate normalisation uses a fixed canonical vocabulary.** Rather than letting the LLM invent canonical forms, a closed vocabulary is defined once and reused at both extraction and normalisation. This guarantees a small, consistent predicate set.

**Graph stored as a `MultiDiGraph`.** Multiple edges between the same pair of nodes are preserved, since the same two entities can appear in different relationships across different papers. Edge access uses `G.out_edges(node, data=True)` / `G.in_edges(node, data=True)` — not `G[u][v]`, which returns the edge-key dict on a multigraph.

**Two-model answer generation.** Query rewriting and triplet extraction use `gpt-4o-mini` for speed and cost. Final answer generation uses `gpt-4.1` for quality.

**Caching throughout.** Each expensive step writes a file to `outputs/` and checks for it on re-run. Delete the relevant file to force recomputation.

## Dependencies

| Package | Role |
|---------|------|
| `pymupdf` / `pymupdf4llm` | PDF → Markdown extraction |
| `llama-index` | Document and chunk abstraction |
| `sentence-transformers` | Dense embeddings (`all-MiniLM-L6-v2`) |
| `scikit-learn` | Cosine similarity |
| `openai` | GPT-4o-mini (extraction, rewriting) and GPT-4.1 (generation) |
| `networkx` | Knowledge graph (`MultiDiGraph`) |
| `pandas` / `numpy` | Data wrangling |
| `matplotlib` | Graph visualisation |
| `tqdm` | Progress bars |
| `python-dotenv` | `.env` secret loading |

## Limitations

This notebook prioritises clarity and educational value over production robustness.

- Entity extraction and junk filtering are heuristic and tuned for this corpus
- No ontology-level entity linking (HGNC, UMLS, etc.)
- Graph stored in memory — not suitable for large corpora
- No re-ranking or hybrid BM25+vector retrieval

## Possible Extensions

- Biomedical entity linking via SciSpaCy + HGNC/UMLS
- BM25 + dense hybrid retrieval
- Graph database backend (Neo4j)
- Community detection for topic clustering
- Streaming answers for interactive use

## License

Provided for research and educational purposes.
