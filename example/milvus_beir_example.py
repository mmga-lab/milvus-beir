from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pymilvus import MilvusClient
from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch
from milvus_beir.retrieval.search.sparse.sparse_search import MilvusSparseSearch
from milvus_beir.retrieval.search.lexical.multi_match_search import MilvusMultiMatchSearch
from milvus_beir.retrieval.search.lexical.bm25_search import MilvusBM25Search
from milvus_beir.retrieval.search.hybrid.bm25_hybrid_search import MilvusBM25DenseHybridSearch
from milvus_beir.retrieval.search.hybrid.sparse_hybrid_search import MilvusSparseDenseHybridSearch

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from ranx import Qrels, Run, compare

dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, "../datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


print("Corpus:", len(corpus))
print("Queries:", len(queries))

milvus_client = MilvusClient(uri="http://10.104.20.192:19530")
# model = MilvusDenseSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
# retriever = EvaluateRetrieval(model)
# dense_results = retriever.retrieve(corpus, queries)
# ndcg, _map, recall, precision = retriever.evaluate(qrels, dense_results, retriever.k_values)
# print("NDCG:", ndcg)
# print("MAP:", _map)
# print("Recall:", recall)
# print("Precision:", precision)
#
#
# model = MilvusSparseSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
# retriever = EvaluateRetrieval(model)
# sparse_results = retriever.retrieve(corpus, queries)
# ndcg, _map, recall, precision = retriever.evaluate(qrels, sparse_results, retriever.k_values)
# print("NDCG:", ndcg)
# print("MAP:", _map)
# print("Recall:", recall)
# print("Precision:", precision)
#
#
#
# model = MilvusBM25DenseHybridSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
# retriever = EvaluateRetrieval(model)
# bm25_dense_hybrid_results = retriever.retrieve(corpus, queries)
# ndcg, _map, recall, precision = retriever.evaluate(qrels, bm25_dense_hybrid_results, retriever.k_values)
# print("NDCG:", ndcg)
# print("MAP:", _map)
# print("Recall:", recall)
# print("Precision:", precision)
#
#
# model = MilvusSparseDenseHybridSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
# retriever = EvaluateRetrieval(model)
# sparse_dense_results = retriever.retrieve(corpus, queries)
# ndcg, _map, recall, precision = retriever.evaluate(qrels, sparse_dense_results, retriever.k_values)
# print("NDCG:", ndcg)
# print("MAP:", _map)
# print("Recall:", recall)
# print("Precision:", precision)
#
#
# model = MilvusMultiMatchSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
# retriever = EvaluateRetrieval(model)
# multi_match_results = retriever.retrieve(corpus, queries)
# ndcg, _map, recall, precision = retriever.evaluate(qrels, multi_match_results, retriever.k_values)
# print("NDCG:", ndcg)
# print("MAP:", _map)
# print("Recall:", recall)
# print("Precision:", precision)
#
#
model = MilvusBM25Search(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
retriever = EvaluateRetrieval(model)
bm25_results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, bm25_results, retriever.k_values)
print("NDCG:", ndcg)
print("MAP:", _map)
print("Recall:", recall)
print("Precision:", precision)


qrels = Qrels(qrels)
# run_dense = Run(dense_results,name="dense")
# run_sparse = Run(sparse_results,name="sparse")
#
# run_multi_match = Run(multi_match_results,name="multi_match")
# run_bm25_dense_hybrid = Run(bm25_dense_hybrid_results,name="bm25_dense_hybrid")
# run_sparse_dense = Run(sparse_dense_results,name="sparse_dense")
run_bm25 = Run(bm25_results, name="bm25")
report = compare(
    qrels=qrels,
    runs=[run_bm25],
    metrics=["ndcg@10", "map@10", "recall@10", "precision@10"],
)
print(report)
