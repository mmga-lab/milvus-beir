from milvus_beir.retrieval.search.milvus import MilvusBaseSearch
import logging
from typing import Dict, List
from tqdm.autonotebook import tqdm
from pymilvus import (connections, Collection, FieldSchema, CollectionSchema,
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


class MilvusBM25DenseHybridSearch(MilvusBaseSearch):
    def __init__(self, collection_name: str, nq: int = 100, nb: int = 2000, uri: str = "localhost",
                 token: str = "19530",
                 initialize: bool = True,
                 clean_up: bool = False,
                 dense_model: BaseEmbeddingFunction = None,
                 bm25_index_params: Dict = None,
                 dense_index_params: Dict = None,
                 dense_search_params: Dict = None
                 ):
        super().__init__(collection_name=collection_name, nq=nq, nb=nb, uri=uri,
                         token=token, search_type="dense",
                         initialize=initialize, clean_up=clean_up,
                         dense_model=dense_model, dense_index_params=dense_index_params,
                         dense_search_params=dense_search_params)

    def handle_search(self, queries: Dict[str, str], top_k: int):
        batch_size = self.nq
        total_rows = len(queries)
        result_list = []
        for start in tqdm(range(0, total_rows, batch_size)):
            end = min(start + batch_size, total_rows)
            batch = queries.iloc[start:end]
            embeddings = batch["emb"].tolist()
            texts = batch["document"].tolist()
            fts_search_req = AnnSearchRequest(
                data=texts,
                anns_field="document_sparse_emb",
                param={},
                limit=top_k,
            )
            dense_search_req = AnnSearchRequest(
                data=embeddings,
                anns_field="dense_emb",
                param={},
                limit=top_k,
            )
            result = self.collection.hybrid_search(
                reqs=[fts_search_req, dense_search_req],
                rerank=WeightedRanker(0.5, 0.5),
                limit=top_k,
                output_fields=["_id"]
            )
            result_list.extend(result)
        result_dict = {}
        for i in range(len(queries)):
            data = {}
            for hit in result_list[i]:
                data[hit.id] = hit.distance
            result_dict[queries[i]["id"]] = data
        return result_dict
