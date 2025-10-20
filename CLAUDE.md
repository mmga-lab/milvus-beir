# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Milvus-BEIR is a Python library that integrates Milvus vector database with BEIR (Benchmarking IR) for efficient information retrieval evaluation. The library provides various search strategies including dense retrieval, sparse retrieval, BM25, and hybrid search approaches.

## Development Commands

### Environment Setup
```bash
# Install dependencies using UV
uv sync

# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run tests with custom Milvus connection
uv run pytest tests/ --milvus-uri="http://localhost:19530" --milvus-token="root:Milvus"

# Run a single test file
uv run pytest tests/test_search.py

# Run a specific test
uv run pytest tests/test_search.py::test_name
```

### Linting and Formatting
```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Run pre-commit on all files
pre-commit run --all-files
```

### CLI Tool
```bash
# Run the milvus-beir CLI tool
uv run milvus-beir --dataset nfcorpus --search-method sparse

# Or after installation
milvus-beir --dataset nfcorpus --search-method sparse
```

## Code Architecture

### Module Structure

The codebase follows a hierarchical structure under `src/milvus_beir/`:

- **`retrieval/search/milvus.py`**: Contains `MilvusBaseSearch`, the abstract base class for all search implementations
  - Handles Milvus client connection and thread-local client management
  - Provides common initialization, cleanup, and batch processing logic
  - All search classes inherit from this base

- **`retrieval/search/dense/`**: Dense vector search using semantic embeddings
  - Uses `milvus-model` library's `SentenceTransformerEmbeddingFunction` by default
  - Schema includes VARCHAR id field and FLOAT_VECTOR field

- **`retrieval/search/sparse/`**: Sparse vector search using models like SPLADE
  - Uses `SpladeEmbeddingFunction` by default
  - Schema includes VARCHAR id field and SPARSE_FLOAT_VECTOR field

- **`retrieval/search/lexical/`**: Lexical search methods
  - `bm25_search.py`: Uses Milvus built-in BM25 Function for text search
  - `multi_match_search.py`: Multi-match search similar to Elasticsearch's best_fields

- **`retrieval/search/colbert/`**: ColBERT multi-vector search
  - `colbert_model.py`: Wrapper for pylate ColBERT model (`mixedbread-ai/mxbai-edge-colbert-v0-17m`)
  - `colbert_search.py`: Multi-vector retrieval using Milvus struct arrays and MAX_SIM_COSINE scoring
  - Each document/query is encoded into multiple token-level embeddings (up to 100 tokens)
  - Uses struct array schema: `ARRAY<STRUCT<clip_embedding: FLOAT_VECTOR>>`

- **`retrieval/search/hybrid/`**: Hybrid search combining multiple strategies
  - `sparse_hybrid_search.py`: Combines sparse and dense vectors using `hybrid_search()` and RRFRanker
  - `bm25_hybrid_search.py`: Combines BM25 and dense vectors

- **`cli/search_cli.py`**: Command-line interface for evaluation using Click
  - Entry point defined in pyproject.toml as `milvus-beir`

- **`utils.py`**: QPS measurement utilities
  - `measure_search_qps_decorator`: Decorator for concurrent search performance testing
  - `QPSMeasurement`: Class for tracking queries, success rates, and calculating QPS

### Common Search Implementation Pattern

All search classes follow this pattern:

1. **Initialization**: Accept `uri`, `token`, `collection_name`, `nq` (query batch size), `nb` (document batch size)
2. **Collection Setup**: `_initialize_collection()` creates schema with appropriate fields and functions
3. **Indexing**: `_index(corpus)` encodes documents in batches and creates indexes
4. **Search**: `search(corpus, queries, top_k)` performs retrieval and returns results as nested dicts
5. **QPS Measurement**: `measure_search_qps()` uses thread pools to measure concurrent search performance

### Threading Model

- **Main client**: `self.milvus_client` used for indexing and single-threaded operations
- **Thread-local clients**: `_get_thread_client()` creates per-thread MilvusClient instances for concurrent QPS testing
- This avoids thread-safety issues with the PyMilvus client

### Hybrid Search Implementation

Hybrid search uses Milvus's built-in reranking:
- Creates multiple `AnnSearchRequest` objects (one per vector field)
- Calls `milvus_client.hybrid_search()` with a ranker (default: `RRFRanker`)
- Returns unified results with reranked scores

### BM25 Implementation

BM25 uses Milvus 2.5+ built-in functionality:
- Schema includes a VARCHAR field with `enable_analyzer=True` and analyzer params
- Adds a `Function` of type `FunctionType.BM25` to generate SPARSE_FLOAT_VECTOR embeddings
- Search is performed on the generated sparse field, not the text field directly

### ColBERT Multi-Vector Implementation

ColBERT uses Milvus struct arrays for multi-vector retrieval:
- Each document is encoded into multiple token-level embeddings (shape: `num_tokens x dim`, where dim is auto-detected)
- Schema uses `ARRAY<STRUCT<clip_embedding: FLOAT_VECTOR>>` to store variable-length token embeddings
- Queries are encoded similarly and converted to `EmbeddingList` format for search
- Uses `MAX_SIM_COSINE` metric: for each query token, find max similarity with all doc tokens, then sum
- Index type: HNSW on `clips[clip_embedding]` field with default params `{"M": 16, "efConstruction": 200}`
- Search params: `{"metric_type": "MAX_SIM_COSINE", "params": {"ef": 1000, "retrieval_ann_ratio": 3.0}}`
  - `ef`: Search quality parameter (higher = better recall but slower)
  - `retrieval_ann_ratio`: Multi-vector retrieval parameter for better accuracy
- Max capacity per document: 100 tokens (configurable)

**Model Configuration:**
- Default model: `mixedbread-ai/mxbai-edge-colbert-v0-17m` (48-dimensional embeddings, 17M parameters)
- Model is configurable via the `model` parameter in `MilvusColBERTSearch.__init__()`
- Can use any ColBERT-compatible model from HuggingFace or local path
- **Embedding dimension is automatically detected** by encoding a test string during initialization
- Custom model example:
  ```python
  from milvus_beir.retrieval.search.colbert.colbert_model import ColBERTEmbeddingFunction

  custom_model = ColBERTEmbeddingFunction(
      model_name_or_path="jinaai/jina-colbert-v1-en",
      batch_size=64
  )
  # Dimension is automatically detected - no manual configuration needed

  search = MilvusColBERTSearch(
      uri="http://localhost:19530",
      token=None,
      collection_name="demo",
      model=custom_model
  )
  ```

## Key Configuration Details

### Dependencies
- Requires `pymilvus>=2.5.0` for BM25, hybrid search, and struct array features
- `numpy<2.0.0` for compatibility
- `torch<=2.2.2` for embedding models
- `milvus-model>=0.2.10` for embedding functions
- `beir>=2.0.0` for BEIR integration
- `pylate>=1.0.0` for ColBERT multi-vector embeddings

### Ruff Configuration
- Line length: 100 characters
- Target Python: 3.8 (though project requires 3.10+)
- Enabled rule sets: E, W, F, I, C, B, UP, N, RUF
- Ignores E501 (line too long) as it's handled by formatter

### Test Fixtures (tests/conftest.py)
- `milvus_uri`: Default "http://localhost:19530", overridable via `--milvus-uri`
- `milvus_token`: Default "root:Milvus", overridable via `--milvus-token`
- `collection_name`: Auto-generated random 8-character string with "test_collection_" prefix
- `test_corpus`: Small 3-document corpus for testing
- `test_queries`: Small 2-query set for testing

## Version Management

Version is defined in `src/milvus_beir/__init__.py` and extracted by Hatchling via:
```toml
[tool.hatch.version]
path = "src/milvus_beir/__init__.py"
```

Update the `__version__` variable in that file to change the package version.
