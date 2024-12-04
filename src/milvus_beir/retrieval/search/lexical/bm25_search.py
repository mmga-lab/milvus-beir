import logging
from typing import Dict, Optional

from pymilvus import DataType, Function, FunctionType, MilvusClient
from tqdm.autonotebook import tqdm

from milvus_beir.retrieval.search.milvus import MilvusBaseSearch

logger = logging.getLogger(__name__)


class MilvusBM25Search(MilvusBaseSearch):
    def __init__(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
        nq: int = 100,
        nb: int = 1000,
        initialize: bool = True,
        clean_up: bool = True,
        analyzer: str = "english",
        bm25_input_field: str = "text",
        bm25_output_field: str = "bm25_sparse",
        metric_type: str = "BM25",
        search_params: Optional[Dict] = None,
        sleep_time: int = 5,
    ):
        self.analyzer = analyzer
        self.bm25_input_field = bm25_input_field
        self.bm25_output_field = bm25_output_field
        self.metric_type = metric_type
        self.search_params = search_params if search_params is not None else {}
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
        analyzer_params = {
            "type": self.analyzer,
        }
        schema.add_field("id", DataType.VARCHAR, max_length=1000, is_primary=True)
        schema.add_field(
            field_name=self.bm25_input_field,
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params=analyzer_params,
        )
        schema.add_field(
            field_name=self.bm25_output_field, datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=[
                self.bm25_input_field
            ],  # Name of the VARCHAR field containing raw text data
            output_field_names=[
                self.bm25_output_field
            ],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        self.milvus_client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

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
            # chunk test with max length of 65536
            texts = [text[:60000] for text in texts]
            ids = corpus_ids[start:end]
            data = [
                {"id": id, self.bm25_input_field: text} for id, text in zip(ids, texts)
            ]
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
        self.milvus_client.flush(self.collection_name)
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name=self.bm25_output_field, metric_type=self.metric_type
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

        batch_size = self.nq
        total_rows = len(queries)
        result_list = []
        for start in tqdm(range(0, total_rows, batch_size)):
            end = min(start + batch_size, total_rows)
            result = self.milvus_client.search(
                collection_name=self.collection_name,
                data=query_texts[start:end],
                anns_field=self.bm25_output_field,
                search_params={},
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
