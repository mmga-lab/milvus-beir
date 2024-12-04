import logging
from typing import Dict, Optional

from pymilvus import DataType, Function, FunctionType, MilvusClient
from tqdm.autonotebook import tqdm

from milvus_beir.retrieval.search.milvus import MilvusBaseSearch

logger = logging.getLogger(__name__)


class MilvusMultiMatchSearch(MilvusBaseSearch):
    def __init__(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
        nq: int = 100,
        nb: int = 1000,
        initialize: bool = True,
        clean_up: bool = True,
        analyzer: str = "english",
        bm25_input_output_mapping: Optional[Dict[str, str]] = None,
        metric_type: str = "BM25",
        search_params: Optional[Dict] = None,
        tie_breaker: float = 0.5,
        sleep_time: int = 5,
    ):
        if bm25_input_output_mapping is None:
            bm25_input_output_mapping = {
                "title": "title_bm25_sparse",
                "text": "text_bm25_sparse",
            }
        self.bm25_input_output_mapping = bm25_input_output_mapping
        self.analyzer = analyzer
        self.metric_type = metric_type
        self.tie_breaker = tie_breaker
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
        for bm25_input_field in self.bm25_input_output_mapping:
            schema.add_field(
                field_name=bm25_input_field,
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params=analyzer_params,
            )

        for bm25_output_field in self.bm25_input_output_mapping.values():
            schema.add_field(field_name=bm25_output_field, datatype=DataType.SPARSE_FLOAT_VECTOR)

        for (
            bm25_input_field,
            bm25_output_field,
        ) in self.bm25_input_output_mapping.items():
            bm25_function = Function(
                name=f"{bm25_input_field}_bm25_emb",  # Function name
                input_field_names=[
                    bm25_input_field
                ],  # Name of the VARCHAR field containing raw text data
                output_field_names=[
                    bm25_output_field
                ],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
                # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)
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
            titles = [doc.get("title", "") for doc in batch]
            texts = [doc.get("text", "") for doc in batch]
            # chunk test with max length of 65536
            titles = [title[:60000] for title in titles]
            texts = [text[:60000] for text in texts]
            ids = corpus_ids[start:end]
            data = [
                {"id": id, "title": title, "text": text}
                for id, title, text in zip(ids, titles, texts)
            ]
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
        self.milvus_client.flush(self.collection_name)
        index_params = self.milvus_client.prepare_index_params()
        for bm25_output_field in self.bm25_input_output_mapping.values():
            index_params.add_index(field_name=bm25_output_field, metric_type=self.metric_type)
            self.milvus_client.create_index(
                collection_name=self.collection_name, index_params=index_params
            )
        self.milvus_client.load_collection(self.collection_name)
        self.index_completed = True
        logger.info("Indexing Completed!")

    def _search_single_field(
        self,
        query_texts: list,
        query_ids: list,
        bm25_output_field: str,
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        """Execute search for a single BM25 field."""
        batch_size = self.nq
        total_rows = len(query_texts)
        result_list = []

        # Batch processing of queries
        for start in tqdm(range(0, total_rows, batch_size)):
            end = min(start + batch_size, total_rows)
            result = self.milvus_client.search(
                collection_name=self.collection_name,
                data=query_texts[start:end],
                anns_field=bm25_output_field,
                search_params={},
                limit=top_k,
                output_fields=["id"],
            )
            result_list.extend(result)

        # Convert results to dictionary format
        result_dict = {}
        for i, query_id in enumerate(query_ids):
            data = {hit["id"]: hit["distance"] for hit in result_list[i]}
            result_dict[query_id] = data

        return result_dict

    def _combine_field_results(
        self,
        multi_result: list,
        query_ids: list,
    ) -> Dict[str, Dict[str, list]]:
        """Combine results from multiple BM25 fields."""
        result_dict = {}
        for query_id in query_ids:
            data = {}
            for result in multi_result:
                for hit_id, distance in result[query_id].items():
                    if hit_id in data:
                        data[hit_id].append(distance)
                    else:
                        data[hit_id] = [distance]
            result_dict[query_id] = data
        return result_dict

    def _apply_fusion(
        self,
        combined_results: Dict[str, Dict[str, list]],
        query_ids: list,
    ) -> Dict[str, Dict[str, float]]:
        """Apply score fusion using tie breaker method."""
        fusion_result = {}
        for query_id in query_ids:
            fusion_result[query_id] = {}
            for hit_id, scores in combined_results[query_id].items():
                scores = sorted(scores, reverse=True)
                fusion_result[query_id][hit_id] = scores[0] + self.tie_breaker * sum(scores[1:])
        return fusion_result

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """Search across multiple BM25 fields and combine results."""
        if self.initialize:
            self._initialize_collection()

        if not self.index_completed:
            self._index(corpus)

        # Prepare query data
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Search each field
        multi_result = []
        for bm25_output_field in self.bm25_input_output_mapping.values():
            field_result = self._search_single_field(
                query_texts, query_ids, bm25_output_field, top_k
            )
            multi_result.append(field_result)

        # Combine results from all fields
        combined_results = self._combine_field_results(multi_result, query_ids)

        # Apply score fusion
        return self._apply_fusion(combined_results, query_ids)
