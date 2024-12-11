#!/usr/bin/env python3

from pathlib import Path

import click
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

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
    "climate-fever": "climate-fever",
    "dbpedia-entity": "dbpedia-entity",
    "fever": "fever",
    "fiqa": "fiqa",
    "hotpotqa": "hotpotqa",
    "nfcorpus": "nfcorpus",
    "nq": "nq",
    "quora": "quora",
    "scidocs": "scidocs",
    "scifact": "scifact",
    "webis-touche2020": "webis-touche2020",
    "trec-covid": "trec-covid",
    "mmarco": "mmarco",
    "cqadupstack/android": "cqadupstack/android",
    "cqadupstack/english": "cqadupstack/english",
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
    "--concurrency-levels",
    "-cl",
    default="1, 2",
    help="Concurrency levels for QPS measurement, comma separated",
)
@click.option("--measure-qps", "-mq", is_flag=True, default=True, help="Whether to measure QPS")
def evaluate(
    dataset, uri, token, search_method, collection_name, nq, nb, concurrency_levels, measure_qps
):
    """CLI tool for evaluating different search methods on BEIR datasets with Milvus."""
    # echo arguments
    click.echo(f"Dataset: {dataset}")
    click.echo(f"URI: {uri}")
    click.echo(f"Token: {token}")
    click.echo(f"Search Method: {search_method}")
    click.echo(f"Collection Name: {collection_name}")
    click.echo(f"Number of Queries: {nq}")
    click.echo(f"Number of Documents: {nb}")
    click.echo(f"Concurrency Levels: {concurrency_levels}")
    click.echo(f"Measure QPS: {measure_qps}\n")
    # Download and load dataset
    if dataset == "cqadupstack":
        subsets = [
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "unix",
            "webmasters",
            "wordpress",
        ]
    else:
        subsets = [None]
    split = "test"
    if dataset == "mmarco":
        split = "dev"  # MS MARCO dataset test split has very few queries
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASETS[dataset]}.zip"
    out_dir = "/tmp/datasets"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    download_data_path = util.download_and_unzip(url, out_dir)
    for subset in subsets:
        if subset is not None:
            data_path = f"{download_data_path}/{subset}"
        else:
            data_path = download_data_path
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        click.echo(f"\nDataset: {dataset}")
        click.echo(f"Corpus size: {len(corpus)}")
        click.echo(f"Number of queries: {len(queries)}")

        # Initialize  search model
        search_class = SEARCH_METHODS[search_method]
        model = search_class(
            uri,
            token,
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

        # Measure QPS only if measure_qps is True
        concurrency_levels = [int(x) for x in concurrency_levels.split(",")]
        if measure_qps:
            qps = model.measure_search_qps(
                corpus, queries, top_k=1000, concurrency_levels=concurrency_levels, test_duration=60
            )
            click.echo(f"QPS: {qps}")


if __name__ == "__main__":
    evaluate()
