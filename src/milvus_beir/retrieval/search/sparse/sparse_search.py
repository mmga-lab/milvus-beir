from milvus_beir.retrieval.search.milvus import MilvusBaseSearch
import torch
import logging
from typing import Dict, List
from tqdm.autonotebook import tqdm
from pymilvus import (connections, Collection, FieldSchema, CollectionSchema, MilvusClient,
                      DataType, utility, FunctionType, Function, AnnSearchRequest, WeightedRanker)
import numpy as np
from pandas import DataFrame
from beir.retrieval.search import BaseSearch
from milvus_model.dense import SentenceTransformerEmbeddingFunction
from milvus_model.sparse import SpladeEmbeddingFunction
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_model.base import BaseEmbeddingFunction

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Dict


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

class MilvusSparseSearch(MilvusBaseSearch):
    def __init__(self,
                 milvus_client: MilvusClient,
                 collection_name: str,
                 nq: int = 100, nb: int = 1000,
                 initialize: bool = True,
                 clean_up: bool = True,
                 model: BaseEmbeddingFunction = SpladeEmbeddingFunction(device=device),
                 vector_field: str = "sparse_embedding",
                 metric_type: str = "IP",
                 search_params: Dict = None
                 ):
        self.model = model
        self.vector_field = vector_field
        self.metric_type = metric_type
        self.search_params = search_params
        super().__init__(milvus_client=milvus_client, collection_name=collection_name, nq=nq, nb=nb,
                         initialize=initialize, clean_up=clean_up)

    def _initialize_collection(self):
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        schema = self.milvus_client.create_schema()
        schema.add_field("id", DataType.VARCHAR, max_length=1000, is_primary=True)
        schema.add_field(self.vector_field, DataType.SPARSE_FLOAT_VECTOR)
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )

    def _index(self, corpus):
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                            reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        for start in tqdm(range(0, len(corpus), self.nb)):
            end = min(start + self.nb, len(corpus))
            batch = corpus[start:end]
            texts = [doc.get("title", "") + " " + doc.get("text", "") for doc in batch]
            embeddings = self.model(texts)
            ids = corpus_ids[start:end]
            data = [{"id": id, self.vector_field: emb} for id, emb in zip(ids, embeddings)]
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=data
            )
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(field_name=self.vector_field, metric_type=self.metric_type)
        self.milvus_client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        self.milvus_client.load_collection(self.collection_name)
        self.index_completed = True
        logger.info("Indexing Completed!")

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               *args,
               **kwargs) -> Dict[str, Dict[str, float]]:

        if self.initialize:
            self._initialize_collection()

        if not self.index_completed:
            self._index(corpus)

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode_queries(query_texts)

        batch_size = self.nq
        total_rows = len(queries)
        result_list = []
        for start in tqdm(range(0, total_rows, batch_size)):
            end = min(start + batch_size, total_rows)
            embeddings = query_embeddings[start:end]
            result = self.milvus_client.search(collection_name=self.collection_name,
                                               data=embeddings,
                                               anns_field=self.vector_field,
                                               search_params={},
                                               limit=top_k,
                                               output_fields=["id"])
            result_list.extend(result)

        result_dict = {}
        for i in range(len(queries)):
            data = {}
            for hit in result_list[i]:
                data[hit["id"]] = hit["distance"]
            result_dict[query_ids[i]] = data
        return result_dict
