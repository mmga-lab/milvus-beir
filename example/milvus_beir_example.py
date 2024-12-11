from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch
from milvus_beir.retrieval.search.hybrid.bm25_hybrid_search import MilvusBM25DenseHybridSearch
from milvus_beir.retrieval.search.hybrid.sparse_hybrid_search import MilvusSparseDenseHybridSearch
from milvus_beir.retrieval.search.lexical.bm25_search import MilvusBM25Search
from milvus_beir.retrieval.search.lexical.multi_match_search import MilvusMultiMatchSearch
from milvus_beir.retrieval.search.sparse.sparse_search import MilvusSparseSearch

dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

print("Corpus:", len(corpus))
print("Queries:", len(queries))

uri = "http://10.104.26.252:19530"
token = None

models = [
    MilvusDenseSearch(uri, token, collection_name="milvus_beir_demo", nq=100, nb=1000),
    MilvusSparseSearch(uri, token, collection_name="milvus_beir_demo", nq=100, nb=1000),
    MilvusBM25DenseHybridSearch(uri, token, collection_name="milvus_beir_demo", nq=100, nb=1000),
    MilvusSparseDenseHybridSearch(uri, token, collection_name="milvus_beir_demo", nq=100, nb=1000),
    MilvusMultiMatchSearch(uri, token, collection_name="milvus_beir_demo", nq=100, nb=1000),
    MilvusBM25Search(uri, token, collection_name="milvus_beir_demo", nq=100, nb=1000),
]

for model in models:
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)
    qps = model.measure_search_qps(
        corpus, queries, top_k=1000, concurrency_levels=[1, 2], test_duration=60
    )
    print("QPS:", qps)
