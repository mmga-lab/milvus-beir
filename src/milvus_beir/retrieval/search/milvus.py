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

class MilvusBaseSearch(BaseSearch):
    def __init__(self,
                 milvus_client: MilvusClient,
                 collection_name: str,
                 initialize: bool = True,
                 clean_up: bool = False,
                 nb: int = 2000,
                 nq: int = 100,
                 ):

        self.milvus_client = milvus_client
        self.collection_name = collection_name
        self.initialize = initialize
        self.clean_up = clean_up
        self.nq = nq
        self.nb = nb
        self.results = {}
        self.index_completed = False




