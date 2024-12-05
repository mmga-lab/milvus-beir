import logging
from typing import Any, Dict, Optional

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.dense import SentenceTransformerEmbeddingFunction
from milvus_model.sparse import SpladeEmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    RRFRanker,
)
from tqdm.autonotebook import tqdm

from milvus_beir.retrieval.search.milvus import MilvusBaseSearch

logger = logging.getLogger(__name__)


def get_default_dense_model() -> BaseEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction()


def get_default_sparse_model() -> BaseEmbeddingFunction:
    return SpladeEmbeddingFunction()


def get_default_ranker():
    return RRFRanker()


class MilvusSparseDenseHybridSearch(MilvusBaseSearch):
    def __init__(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
        nq: int = 100,
        nb: int = 1000,
        initialize: bool = True,
        clean_up: bool = True,
        dense_model: BaseEmbeddingFunction = None,
        sparse_model: BaseEmbeddingFunction = None,
        dense_vector_field: str = "dense_embedding",
        sparse_vector_field: str = "sparse_embedding",
        dense_metric_type: str = "COSINE",
        sparse_metric_type: str = "IP",
        dense_search_params: Optional[Dict] = None,
        sparse_search_params: Optional[Dict] = None,
        ranker: Any = None,
    ):
        self.dense_model = dense_model if dense_model is not None else get_default_dense_model()
        self.sparse_model = sparse_model if sparse_model is not None else get_default_sparse_model()
        self.dense_vector_field = dense_vector_field
        self.sparse_vector_field = sparse_vector_field
        self.dense_metric_type = dense_metric_type
        self.sparse_metric_type = sparse_metric_type
        self.dense_search_params = dense_search_params if dense_search_params is not None else {}
        self.sparse_search_params = sparse_search_params if sparse_search_params is not None else {}
        self.ranker = ranker if ranker is not None else get_default_ranker()

        super().__init__(
            milvus_client=milvus_client,
            collection_name=collection_name,
            nq=nq,
            nb=nb,
            initialize=initialize,
            clean_up=clean_up,
        )

    def _initialize_collection(self):
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        schema = self.milvus_client.create_schema()
        schema.add_field("id", DataType.VARCHAR, max_length=1000, is_primary=True)
        schema.add_field(self.dense_vector_field, DataType.FLOAT_VECTOR, dim=self.dense_model.dim)
        schema.add_field(self.sparse_vector_field, DataType.SPARSE_FLOAT_VECTOR)
        self.milvus_client.create_collection(collection_name=self.collection_name, schema=schema)

    def _index(self, corpus):
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [corpus[cid] for cid in corpus_ids]
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        for start in tqdm(range(0, len(corpus), self.nb)):
            end = min(start + self.nb, len(corpus))
            batch = corpus[start:end]
            texts = [doc.get("title", "") + " " + doc.get("text", "") for doc in batch]
            dense_embeddings = self.dense_model(texts)
            sparse_embeddings = self.sparse_model(texts)
            ids = corpus_ids[start:end]
            data = [
                {
                    "id": id,
                    self.dense_vector_field: dense_emb,
                    self.sparse_vector_field: sparse_emb,
                }
                for id, dense_emb, sparse_emb in zip(ids, dense_embeddings, sparse_embeddings)
            ]
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
        self.milvus_client.flush(self.collection_name)
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name=self.dense_vector_field, metric_type=self.dense_metric_type
        )

        self.milvus_client.create_index(
            collection_name=self.collection_name, index_params=index_params
        )
        index_params.add_index(
            field_name=self.sparse_vector_field, metric_type=self.sparse_metric_type
        )
        self.milvus_client.create_index(
            collection_name=self.collection_name, index_params=index_params
        )

        self.milvus_client.load_collection(self.collection_name)
        self.index_completed = True
        logger.info("Indexing Completed!")

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if self.initialize:
            self._initialize_collection()

        if not self.index_completed:
            self._index(corpus)

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        dense_query_embeddings = self.dense_model.encode_queries(query_texts)
        sparse_query_embeddings = self.sparse_model.encode_queries(query_texts)

        batch_size = self.nq
        total_rows = len(queries)
        result_list = []
        for start in tqdm(range(0, total_rows, batch_size)):
            end = min(start + batch_size, total_rows)
            dense_embeddings = dense_query_embeddings[start:end]
            sparse_embeddings = sparse_query_embeddings[start:end]

            dense_rqs = AnnSearchRequest(
                data=dense_embeddings,
                anns_field=self.dense_vector_field,
                param=self.dense_search_params,
                limit=top_k,
            )
            sparse_rqs = AnnSearchRequest(
                data=sparse_embeddings,
                anns_field=self.sparse_vector_field,
                param=self.sparse_search_params,
                limit=top_k,
            )

            result = self.milvus_client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_rqs, sparse_rqs],
                ranker=self.ranker,
                limit=top_k,
                output_fields=["id"],
            )
            result_list.extend(result)

        result_dict = {}
        for i in range(len(queries)):
            data = {}
            for hit in result_list[i]:
                data[hit["id"]] = hit["distance"]
            result_dict[query_ids[i]] = data

        return result_dict
