# Milvus BEIR

A Python library that integrates Milvus vector database with BEIR (Benchmarking IR) for efficient information retrieval and evaluation. This library provides various search strategies including dense retrieval, sparse retrieval, and hybrid search approaches.

## Features

- Multiple search strategies:
  - Dense Vector Search
  - Sparse Vector Search
  - BM25 Search
  - Hybrid Search (BM25 + Dense)
  - Multi-Match Search
- Seamless integration with BEIR datasets and evaluation metrics
- Easy-to-use API for retrieval and evaluation
- Compatible with Milvus 2.x

## Installation

```bash
pip install milvus-beir
```

## Prerequisites

- Python >= 3.8
- Running Milvus instance (2.5.0 or higher)

## Quick Start

Here's a simple example of how to use milvus-beir:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pymilvus import MilvusClient
from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch

# Download BEIR dataset
dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "/tmp/datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Initialize Milvus client and search model
milvus_client = MilvusClient(uri="http://localhost:19530")
model = MilvusDenseSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)

# Perform retrieval and evaluation
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
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

### Hybrid Search
Combines different search strategies for better results.
```python
from milvus_beir.retrieval.search.hybrid.bm25_hybrid_search import MilvusBM25DenseHybridSearch
from milvus_beir.retrieval.search.hybrid.sparse_hybrid_search import MilvusSparseDenseHybridSearch
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/milvus-beir.git
cd milvus-beir

# Install dependencies using PDM
pdm install
```

### Running Tests

```bash
pdm run pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.
