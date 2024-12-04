import logging
from pymilvus import MilvusClient
from beir.retrieval.search import BaseSearch

logger = logging.getLogger(__name__)


class MilvusBaseSearch(BaseSearch):
    def __init__(
        self,
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
