from milvus_beir.retrieval.search.dense.dense_search import MilvusDenseSearch
from milvus_beir.retrieval.search.hybrid.bm25_hybrid_search import MilvusBM25DenseHybridSearch
from milvus_beir.retrieval.search.hybrid.sparse_hybrid_search import MilvusSparseDenseHybridSearch
from milvus_beir.retrieval.search.lexical.bm25_search import MilvusBM25Search
from milvus_beir.retrieval.search.lexical.multi_match_search import MilvusMultiMatchSearch
from milvus_beir.retrieval.search.sparse.sparse_search import MilvusSparseSearch


def test_dense_search(milvus_client, collection_name, test_corpus, test_queries):
    searcher = MilvusDenseSearch(
        milvus_client=milvus_client,
        collection_name=collection_name,
    )

    results = searcher.search(test_corpus, test_queries, top_k=2)

    # Verify basic structure of results
    assert isinstance(results, dict)
    assert len(results) == len(test_queries)
    for qid in test_queries:
        assert qid in results
        assert isinstance(results[qid], dict)
        assert len(results[qid]) <= 2  # top_k=2


def test_sparse_search(milvus_client, collection_name, test_corpus, test_queries):
    searcher = MilvusSparseSearch(milvus_client=milvus_client, collection_name=collection_name)

    results = searcher.search(test_corpus, test_queries, top_k=2)

    assert isinstance(results, dict)
    assert len(results) == len(test_queries)
    for qid in test_queries:
        assert qid in results
        assert isinstance(results[qid], dict)
        assert len(results[qid]) <= 2


def test_sparse_hybrid_search(milvus_client, collection_name, test_corpus, test_queries):
    searcher = MilvusSparseDenseHybridSearch(
        milvus_client=milvus_client,
        collection_name=collection_name,
    )

    results = searcher.search(test_corpus, test_queries, top_k=2)

    assert isinstance(results, dict)
    assert len(results) == len(test_queries)
    for qid in test_queries:
        assert qid in results
        assert isinstance(results[qid], dict)
        assert len(results[qid]) <= 2


def test_bm25_hybrid_search(milvus_client, collection_name, test_corpus, test_queries):
    searcher = MilvusBM25DenseHybridSearch(
        milvus_client=milvus_client,
        collection_name=collection_name,
    )

    results = searcher.search(test_corpus, test_queries, top_k=2)

    assert isinstance(results, dict)
    assert len(results) == len(test_queries)
    for qid in test_queries:
        assert qid in results
        assert isinstance(results[qid], dict)
        assert len(results[qid]) <= 2


def test_bm25_search(milvus_client, collection_name, test_corpus, test_queries):
    searcher = MilvusBM25Search(milvus_client=milvus_client, collection_name=collection_name)

    results = searcher.search(test_corpus, test_queries, top_k=2)

    assert isinstance(results, dict)
    assert len(results) == len(test_queries)
    for qid in test_queries:
        assert qid in results
        assert isinstance(results[qid], dict)
        assert len(results[qid]) <= 2


def test_multi_match_search(milvus_client, collection_name, test_corpus, test_queries):
    searcher = MilvusMultiMatchSearch(milvus_client=milvus_client, collection_name=collection_name)

    results = searcher.search(test_corpus, test_queries, top_k=2)

    assert isinstance(results, dict)
    assert len(results) == len(test_queries)
    for qid in test_queries:
        assert qid in results
        assert isinstance(results[qid], dict)
        assert len(results[qid]) <= 2
