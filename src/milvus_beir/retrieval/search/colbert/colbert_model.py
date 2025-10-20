import logging
from typing import List

import numpy as np
from pylate import models

logger = logging.getLogger(__name__)


class ColBERTEmbeddingFunction:
    """
    ColBERT embedding function wrapper for multi-vector retrieval.

    This class wraps the pylate ColBERT model to provide multi-vector embeddings
    for documents and queries. Each document/query is encoded into multiple token-level
    embeddings, which are then used for MAX_SIM scoring in Milvus.

    The embedding dimension is automatically detected during initialization by encoding
    a test string, so this class works with any ColBERT model regardless of its output
    dimension (e.g., 48-dim for mxbai-edge-colbert, 128-dim for ColBERTv2, etc.).
    """

    def __init__(
        self,
        model_name_or_path: str = "mixedbread-ai/mxbai-edge-colbert-v0-17m",
        batch_size: int = 32,
    ):
        """
        Initialize the ColBERT embedding function.

        Args:
            model_name_or_path: HuggingFace model name or path
            batch_size: Batch size for encoding
        """
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        logger.info(f"Loading ColBERT model: {model_name_or_path}")
        self.model = models.ColBERT(model_name_or_path=model_name_or_path)
        logger.info("ColBERT model loaded successfully")

        # Dynamically detect embedding dimension by encoding a test string
        logger.info("Detecting embedding dimension...")
        self._dim = self._detect_dimension()
        logger.info(f"Detected embedding dimension: {self._dim}")

    def _detect_dimension(self) -> int:
        """
        Detect the embedding dimension by encoding a test string.

        Returns:
            The embedding dimension of the model
        """
        test_text = "test"
        test_embedding = self.model.encode(
            [test_text],
            batch_size=1,
            is_query=False,
            show_progress_bar=False,
        )
        # test_embedding is a list with one 2D array of shape (num_tokens, dim)
        return test_embedding[0].shape[1]

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    def encode(
        self, texts: List[str], batch_size: int|None = None, show_progress_bar: bool = True
    ) -> List[np.ndarray]:
        """
        Encode documents into multi-vector embeddings.

        Args:
            texts: List of document texts to encode
            batch_size: Batch size for encoding (defaults to self.batch_size)
            show_progress_bar: Whether to show progress bar

        Returns:
            List of 2D numpy arrays, where each array has shape (num_tokens, dim)
        """
        if batch_size is None:
            batch_size = self.batch_size

        logger.info(f"Encoding {len(texts)} documents with ColBERT...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=show_progress_bar,
        )

        # embeddings is a list of 2D arrays (one per document)
        # Each array has shape (num_tokens, dim)
        return embeddings

    def encode_queries(
        self, queries: List[str], batch_size: int|None = None, show_progress_bar: bool = True
    ) -> List[np.ndarray]:
        """
        Encode queries into multi-vector embeddings.

        Args:
            queries: List of query texts to encode
            batch_size: Batch size for encoding (defaults to self.batch_size)
            show_progress_bar: Whether to show progress bar

        Returns:
            List of 2D numpy arrays, where each array has shape (num_tokens, dim)
        """
        if batch_size is None:
            batch_size = self.batch_size

        logger.info(f"Encoding {len(queries)} queries with ColBERT...")
        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            is_query=True,
            show_progress_bar=show_progress_bar,
        )

        # embeddings is a list of 2D arrays (one per query)
        # Each array has shape (num_tokens, dim)
        return embeddings

    def __call__(self, texts: List[str]) -> List[np.ndarray]:
        """
        Convenience method for encoding documents.

        Args:
            texts: List of document texts to encode

        Returns:
            List of 2D numpy arrays with multi-vector embeddings
        """
        return self.encode(texts)
