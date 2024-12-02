from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pymilvus import MilvusClient
from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch



from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


print("Corpus:", len(corpus))
print("Queries:", len(queries))

milvus_client = MilvusClient(uri="http://10.104.20.192:19530")
model = MilvusDenseSearch(milvus_client, collection_name="milvus_beir_demo", nq=100, nb=1000)
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
