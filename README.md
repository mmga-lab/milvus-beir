# Milvus BEIR

A Python library that integrates Milvus vector database with BEIR (Benchmarking IR) for efficient information retrieval and evaluation. This library provides various search strategies including dense retrieval, sparse retrieval, and hybrid search approaches.

## Features

- Multiple search strategies:
  - Dense Vector Search
  - Sparse Vector Search
  - BM25 Search
  - Hybrid Search (BM25 + Dense, Sparse + Dense)
  - Multi-Match Search
- Seamless integration with BEIR datasets and evaluation metrics
- Easy-to-use API for retrieval and evaluation
- Built-in performance measurement (QPS)
- Compatible with Milvus 2.5.x

## Installation

```bash
pip install milvus-beir
```

## Prerequisites

- Python >= 3.10
- Running Milvus instance (2.5.0 or higher)

## Quick Start

Here's a comprehensive example of how to use milvus-beir for search and evaluation:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch

# Download and load BEIR dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Initialize Milvus search model
uri = "http://localhost:19530"
model = MilvusDenseSearch(
    uri, 
    token=None,  # Optional authentication token
    collection_name="milvus_beir_demo",
    nq=100,  # Number of queries to process in parallel
    nb=1000  # Number of documents to process in parallel
)

# Perform retrieval and evaluation
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

# Calculate standard metrics
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# Measure search performance (QPS)
qps = model.measure_search_qps(
    corpus, 
    queries, 
    top_k=1000,
    concurrency_levels=[1, 2],
    test_duration=60
)
```

## Search Strategies

### Dense Vector Search
Uses dense embeddings for semantic search.
```python
from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch
```

### Sparse Vector Search
Implements sparse vector retrieval.
```python
from milvus_beir.retrieval.search.sparse.sparse_search import MilvusSparseSearch
```

### BM25 Search
Traditional lexical search using BM25 algorithm.
```python
from milvus_beir.retrieval.search.lexical.bm25_search import MilvusBM25Search
```

### Multi-Match Search
Implements a multi-match search strategy similar to Elasticsearch's multi-match with best_fields type.
```python
from milvus_beir.retrieval.search.multi_match.multi_match_search import MilvusMultiMatchSearch
```

### Hybrid Search
Combines different search strategies for better results.
```python
from milvus_beir.retrieval.search.hybrid.bm25_hybrid_search import MilvusBM25DenseHybridSearch
from milvus_beir.retrieval.search.hybrid.sparse_hybrid_search import MilvusSparseDenseHybridSearch
```

## Command Line Interface

The package includes a powerful command-line interface for evaluating different search methods:

```bash
# Basic usage
milvus-beir --dataset nfcorpus --search-method sparse

# Full options
milvus-beir \
    --dataset nfcorpus \
    --uri "http://localhost:19530" \
    --token "your_token" \
    --search-method bm25 \
    --collection-name "my_collection" \
    --nq 100 \
    --nb 1000 \
    --concurrency-levels "1,2" \
    --measure-qps
```

Available options:
- `--dataset, -d`: Dataset name to evaluate on (required)
  - Supported datasets: climate-fever, dbpedia-entity, fever, fiqa, hotpotqa, nfcorpus, nq, quora, scidocs, scifact, webis-touche2020, trec-covid, mmarco, cqadupstack/android, cqadupstack/english
- `--uri, -u`: Milvus server URI (default: http://localhost:19530)
- `--token, -t`: Authentication token for Milvus (optional)
- `--search-method, -m`: Search method to use (required)
  - Available methods: dense, sparse, sparse_hybrid, bm25_hybrid, multi_match, bm25
- `--collection-name, -c`: Milvus collection name (optional)
- `--nq`: Number of queries to process in parallel (default: 100)
- `--nb`: Number of documents to process in parallel (default: 1000)
- `--concurrency-levels`: Comma-separated list of concurrency levels for QPS measurement (default: "1,2")
- `--measure-qps`: Flag to enable QPS measurement (default: True)

The CLI tool will:
1. Download and load the specified BEIR dataset
2. Initialize the selected search method
3. Perform retrieval and evaluation
4. Calculate and display standard metrics (NDCG@k, MAP@k, Recall@k, Precision@k)
5. Measure and report search performance in QPS (Queries Per Second)

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/milvus-beir.git
cd milvus-beir

# Install dependencies using PDM
pdm install
pre-commit install
```

### Running Tests

```bash
pdm run pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.