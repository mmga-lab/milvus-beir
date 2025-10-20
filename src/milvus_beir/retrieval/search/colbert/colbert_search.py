import logging
import random
import time
from typing import Dict, Optional

from pymilvus import DataType
from pymilvus.client.embedding_list import EmbeddingList
from tqdm.autonotebook import tqdm

from milvus_beir.retrieval.search.colbert.colbert_model import ColBERTEmbeddingFunction
from milvus_beir.retrieval.search.milvus import MilvusBaseSearch
from milvus_beir.utils import DEFAULT_CONCURRENCY_LEVELS, measure_search_qps_decorator

logger = logging.getLogger(__name__)


def get_default_colbert_model() -> ColBERTEmbeddingFunction:
    """Get the default ColBERT model."""
    return ColBERTEmbeddingFunction()


class MilvusColBERTSearch(MilvusBaseSearch):
    """
    Milvus ColBERT multi-vector search implementation.

    This class implements ColBERT retrieval using Milvus struct arrays for multi-vector
    storage and MAX_SIM_COSINE metric for scoring. Each document is encoded into multiple
    token-level embeddings, which are stored as a struct array in Milvus.

    MAX_SIM scoring: For each query token, find the maximum similarity with all document
    tokens, then sum these max similarities across query tokens.
    """

    def __init__(
        self,
        uri: str,
        token: str | None,
        collection_name: str,
        nq: int = 100,
        nb: int = 1000,
        initialize: bool = True,
        clean_up: bool = True,
        model: ColBERTEmbeddingFunction = None,
        vector_field: str = "clip_embedding",
        struct_array_field: str = "clips",
        metric_type: str = "MAX_SIM_COSINE",
        max_capacity: int = 100,
        search_params: Optional[Dict] = None,
        index_params: Optional[Dict] = None,
        sleep_time: int = 5,
    ):
        """
        Initialize ColBERT search.

        Args:
            uri: Milvus server URI
            token: Authentication token
            collection_name: Name of the collection
            nq: Number of queries to process in parallel
            nb: Number of documents to process in parallel
            initialize: Whether to initialize collection
            clean_up: Whether to clean up collection on destruction
            model: ColBERT embedding function
            vector_field: Name of the vector field within struct
            struct_array_field: Name of the struct array field
            metric_type: Metric type for search (MAX_SIM_COSINE)
            max_capacity: Maximum capacity of struct array
            search_params: Search parameters (e.g., {"ef": 200, "retrieval_ann_ratio": 3.0})
            index_params: Index parameters (e.g., {"M": 16, "efConstruction": 200})
            sleep_time: Sleep time after indexing
        """
        self.model = model if model is not None else get_default_colbert_model()
        self.vector_field = vector_field
        self.struct_array_field = struct_array_field
        self.metric_type = metric_type
        self.max_capacity = max_capacity
        self.search_params = (
            search_params
            if search_params is not None
            else {"metric_type": "MAX_SIM_COSINE", "params": {"ef": 200, "retrieval_ann_ratio": 3.0}}
        )
        self.index_params = (
            index_params if index_params is not None else {"M": 16, "efConstruction": 200}
        )
        self.sleep_time = sleep_time
        self.query_embeddings = []

        super().__init__(
            uri=uri,
            token=token,
            collection_name=collection_name,
            nq=nq,
            nb=nb,
            initialize=initialize,
            clean_up=clean_up,
        )

    def _initialize_collection(self):
        """Initialize collection with struct array schema for multi-vector storage."""
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

        logger.info(f"Creating collection {self.collection_name} with struct array schema...")

        # Create main schema
        schema = self.milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=1000, is_primary=True)

        # Create struct schema for multi-vector embeddings
        struct_schema = self.milvus_client.create_struct_field_schema()
        struct_schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.model.dim)

        # Add struct array field to main schema
        schema.add_field(
            self.struct_array_field,
            datatype=DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=self.max_capacity,
        )

        # Create collection
        self.milvus_client.create_collection(collection_name=self.collection_name, schema=schema)
        logger.info(f"Collection {self.collection_name} created successfully")

    def _index(self, corpus):
        """
        Index corpus documents with ColBERT multi-vector embeddings.

        Each document is encoded into multiple token embeddings, which are stored
        as a struct array in Milvus.
        """
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus_data = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches with ColBERT... Warning: This might take a while!")
        for start in tqdm(range(0, len(corpus_data), self.nb)):
            end = min(start + self.nb, len(corpus_data))
            batch = corpus_data[start:end]
            texts = [doc.get("title", "") + " " + doc.get("text", "") for doc in batch]

            # Encode with ColBERT - returns list of 2D arrays (num_tokens, dim)
            embeddings_list = self.model.encode(texts, show_progress_bar=False)

            # Convert to struct array format
            ids = corpus_ids[start:end]
            data = []
            for i, doc_id in enumerate(ids):
                doc_embeddings = embeddings_list[i]  # Shape: (num_tokens, dim)

                # Convert to list of structs
                clips = []
                num_tokens = min(doc_embeddings.shape[0], self.max_capacity)
                for j in range(num_tokens):
                    clip_struct = {self.vector_field: doc_embeddings[j].tolist()}
                    clips.append(clip_struct)

                row = {"id": doc_id, self.struct_array_field: clips}
                data.append(row)

            self.milvus_client.insert(collection_name=self.collection_name, data=data)

        self.milvus_client.flush(self.collection_name)
        logger.info("Data insertion completed, creating index...")

        # Create index on struct array vector field
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name=f"{self.struct_array_field}[{self.vector_field}]",
            index_name="colbert_vector_index",
            index_type="HNSW",
            metric_type=self.metric_type,
            params=self.index_params,
        )

        self.milvus_client.create_index(collection_name=self.collection_name, index_params=index_params)
        logger.info("Index created successfully, loading collection...")

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
        """
        Perform ColBERT multi-vector search.

        Args:
            corpus: Document corpus
            queries: Query dict
            top_k: Number of top results to return

        Returns:
            Dict mapping query IDs to dicts of document IDs and scores
        """
        if self.initialize:
            self._initialize_collection()

        if not self.index_completed:
            self._index(corpus)

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Encode queries with ColBERT
        logger.info("Encoding queries with ColBERT...")
        query_embeddings_list = self.model.encode_queries(query_texts, show_progress_bar=True)

        batch_size = self.nq
        total_rows = len(queries)
        result_list = []

        logger.info("Performing searches...")
        for start in tqdm(range(0, total_rows, batch_size)):
            end = min(start + batch_size, total_rows)

            # Convert query embeddings to EmbeddingList format
            batch_query_data = []
            for i in range(start, end):
                query_emb = query_embeddings_list[i]  # Shape: (num_query_tokens, dim)
                embedding_list = EmbeddingList()
                for token_emb in query_emb:
                    embedding_list.add(token_emb.tolist())
                batch_query_data.append(embedding_list)

            # Perform search
            result = self.milvus_client.search(
                collection_name=self.collection_name,
                data=batch_query_data,
                anns_field=f"{self.struct_array_field}[{self.vector_field}]",
                search_params=self.search_params,
                limit=top_k,
                output_fields=["id"],
            )
            result_list.extend(result)

        # Convert results to dict format
        result_dict = {}
        for i in range(len(queries)):
            data = {}
            for hit in result_list[i]:
                data[hit["id"]] = hit["distance"]
            result_dict[query_ids[i]] = data

        return result_dict

    def measure_search_qps(
        self, corpus, queries, top_k=1000, concurrency_levels=None, test_duration=60
    ):
        """
        Measure search QPS with different concurrency levels.

        Args:
            corpus: Document corpus
            queries: Query dict
            top_k: Number of top results to return
            concurrency_levels: List of concurrency levels to test
            test_duration: Duration of each test in seconds

        Returns:
            List of (concurrency_level, qps) tuples
        """
        if concurrency_levels is None:
            concurrency_levels = DEFAULT_CONCURRENCY_LEVELS

        @measure_search_qps_decorator(
            concurrency_levels, test_duration=test_duration, max_threads=None
        )
        def _single_search(top_k):
            """Perform a single search operation."""
            # Randomly select a query embedding
            random_idx = random.randint(0, len(self.query_embeddings) - 1)
            query_emb = self.query_embeddings[random_idx]

            # Convert to EmbeddingList
            embedding_list = EmbeddingList()
            for token_emb in query_emb:
                embedding_list.add(token_emb.tolist())

            try:
                client = self._get_thread_client()
                result = client.search(
                    collection_name=self.collection_name,
                    data=[embedding_list],
                    anns_field=f"{self.struct_array_field}[{self.vector_field}]",
                    search_params=self.search_params,
                    limit=top_k,
                    output_fields=["id"],
                )
                return result
            except Exception as e:
                logger.error(f"Search error: {e!s}")
                return None

        # Ensure index is completed
        if not self.index_completed:
            self._index(corpus)
            time.sleep(self.sleep_time)

        # Pre-encode queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        logger.info("Pre-encoding queries for QPS measurement...")
        self.query_embeddings = self.model.encode_queries(query_texts, show_progress_bar=True)

        # Run QPS measurement
        res = _single_search(top_k)
        return res
