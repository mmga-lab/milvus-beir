#!/usr/bin/env python3

import click
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pymilvus import MilvusClient

from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch
from milvus_beir.retrieval.search.hybrid.bm25_hybrid_search import MilvusBM25DenseHybridSearch
from milvus_beir.retrieval.search.hybrid.sparse_hybrid_search import MilvusSparseDenseHybridSearch
from milvus_beir.retrieval.search.lexical.bm25_search import MilvusBM25Search
from milvus_beir.retrieval.search.lexical.multi_match_search import MilvusMultiMatchSearch
from milvus_beir.retrieval.search.sparse.sparse_search import MilvusSparseSearch

SEARCH_METHODS = {
    "dense": MilvusDenseSearch,
    "sparse": MilvusSparseSearch,
    "sparse_hybrid": MilvusSparseDenseHybridSearch,
    "bm25_hybrid": MilvusBM25DenseHybridSearch,
    "multi_match": MilvusMultiMatchSearch,
    "bm25": MilvusBM25Search,
}

DATASETS = {
    "arguana": "arguana",
    "climate-fever": "climate-fever",
    "cqadupstack": "cqadupstack",
    "dbpedia-entity": "dbpedia-entity",
    "fever": "fever",
    "fiqa": "fiqa",
    "germanquad": "germanquad",
    "hotpotqa": "hotpotqa",
    "mmarco": "mmarco",
    "mrtydi": "mrtydi",
    "msmarco-v2": "msmarco-v2",
    "msmarco": "msmarco",
    "nfcorpus": "nfcorpus",
    "nq-train": "nq-train",
    "nq": "nq",
    "quora": "quora",
    "scidocs": "scidocs",
    "scifact": "scifact",
    "trec-covid-beir": "trec-covid-beir",
    "trec-covid-v2": "trec-covid-v2",
    "trec-covid": "trec-covid",
    "vihealthqa": "vihealthqa",
    "webis-touche2020": "webis-touche2020",
}


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(list(DATASETS.keys())),
    required=True,
    help="Dataset name to evaluate on",
)
@click.option("--uri", "-u", default="http://localhost:19530", help="Milvus server URI")
@click.option("--token", "-t", default=None, help="Authentication token for Milvus")
@click.option(
    "--search-method",
    "-m",
    type=click.Choice(list(SEARCH_METHODS.keys())),
    required=True,
    help="Search method to use",
)
@click.option("--collection-name", "-c", default=None, help="Milvus collection name")
@click.option("--nq", default=100, help="Number of queries to process in parallel")
@click.option("--nb", default=1000, help="Number of documents to process in parallel")
@click.option(
    "--split",
    default="test",
    type=click.Choice(["train", "test", "dev"]),
    help="Dataset split to evaluate on",
)
def evaluate(dataset, uri, token, search_method, collection_name, nq, nb, split):
    """CLI tool for evaluating different search methods on BEIR datasets with Milvus."""
    # Download and load dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASETS[dataset]}.zip"
    data_path = util.download_and_unzip(url, "/tmp/datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    click.echo(f"\nDataset: {dataset}")
    click.echo(f"Corpus size: {len(corpus)}")
    click.echo(f"Number of queries: {len(queries)}")

    # Initialize Milvus client and search model
    milvus_client = MilvusClient(uri=uri, token=token)
    search_class = SEARCH_METHODS[search_method]
    model = search_class(
        milvus_client,
        collection_name=collection_name or f"beir_{dataset}_{search_method}",
        nq=nq,
        nb=nb,
    )

    # Perform evaluation
    click.echo(f"\nEvaluating {search_method} search method...")
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    # Print results
    click.echo("\nEvaluation Results:")
    click.echo(f"NDCG@k: {ndcg}")
    click.echo(f"MAP@k: {_map}")
    click.echo(f"Recall@k: {recall}")
    click.echo(f"Precision@k: {precision}")


if __name__ == "__main__":
    evaluate()
