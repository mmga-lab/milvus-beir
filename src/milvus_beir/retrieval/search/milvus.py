import logging
import threading

from beir.retrieval.search import BaseSearch
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class MilvusBaseSearch(BaseSearch):
    def __init__(
        self,
        uri: str,
        token: str | None,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        nb: int = 2000,
        nq: int = 100,
    ):
        self.uri = uri
        self.token = token
        self.milvus_client = MilvusClient(uri=uri, token=token)
        self.collection_name = collection_name
        self.initialize = initialize
        self.clean_up = clean_up
        self.nq = nq
        self.nb = nb
        self.results = {}
        self.index_completed = False
        self._thread_local = threading.local()

    def _get_thread_client(self):
        """ """
        if not hasattr(self._thread_local, "client"):
            self._thread_local.client = MilvusClient(uri=self.uri, token=self.token)
        return self._thread_local.client
