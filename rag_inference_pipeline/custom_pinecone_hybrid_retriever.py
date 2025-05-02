"""Taken from: https://docs.pinecone.io/docs/hybrid-search"""

import hashlib
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import pre_init
import numpy as np
from pydantic import ConfigDict


def hash_text(text: str) -> str:
    """Hash a text using SHA256.

    Args:
        text: Text to hash.

    Returns:
        Hashed text.
    """
    return str(hashlib.sha256(text.encode("utf-8")).hexdigest())


def create_index(
    text_contents: List[str],
    index: Any,
    embeddings: Embeddings,
    sparse_encoder: Any,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None,
    namespace: Optional[str] = None,
    text_key: str = "text",
) -> None:
    """Create an index from a list of contexts.

    It modifies the index argument in-place!

    Args:
        contexts: List of contexts to embed.
        index: Index to use.
        embeddings: Embeddings model to use.
        sparse_encoder: Sparse encoder to use.
        ids: List of ids to use for the documents.
        metadatas: List of metadata to use for the documents.
        namespace: Namespace value for index partition.
    """
    batch_size = 32
    _iterator = range(0, len(text_contents), batch_size)
    try:
        from tqdm.auto import tqdm

        _iterator = tqdm(_iterator)
    except ImportError:
        pass

    if ids is None:
        # create unique ids using hash of the text
        ids = [hash_text(text_content) for text_content in text_contents]

    for i in _iterator:
        # find end of batch
        i_end = min(i + batch_size, len(text_contents))
        # extract batch
        text_contents_batch = text_contents[i:i_end]
        batch_ids = ids[i:i_end]
        metadata_batch = (
            metadatas[i:i_end] if metadatas else [{} for _ in text_contents_batch]
        )
        # add context passages as metadata
        meta = [
            {text_key: doc_content, **metadata}
            for doc_content, metadata in zip(text_contents_batch, metadata_batch)
        ]

        # create dense vectors
        dense_embeds = embeddings.embed_documents(text_contents_batch)
        # create sparse vectors
        sparse_embeds = sparse_encoder.encode_documents(text_contents_batch)
        for s in sparse_embeds:
            s["values"] = [float(s1) for s1 in s["values"]]

        vectors = []
        # loop through the data and create dictionaries for upserts
        for doc_id, sparse, dense, metadata in zip(
            batch_ids, sparse_embeds, dense_embeds, meta
        ):
            vectors.append(
                {
                    "id": doc_id,
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata,
                }
            )

        # upload the documents to the new hybrid index
        index.upsert(vectors, namespace=namespace)


class PineconeHybridSearchRetriever(BaseRetriever):
    """`Pinecone Hybrid Search` retriever."""

    embeddings: Embeddings
    """Embeddings model to use."""
    """description"""
    sparse_encoder: Any = None
    """Sparse encoder to use."""
    index: Any = None
    """Pinecone index to use."""
    top_k: int = 4
    """Number of documents to return."""
    alpha: float = 0.5
    """Alpha value for hybrid search."""
    namespace: Optional[str] = None
    """Namespace value for index partition."""
    text_content_key: str = "text"
    """Key to use for text content in the metadata. 
    Original PineconeHybridSearchRetriever class uses 'context' as the key for content in the metadata rather than 'text' key which is more in use."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        create_index(
            texts,
            self.index,
            self.embeddings,
            self.sparse_encoder,
            ids=ids,
            metadatas=metadatas,
            namespace=namespace,
        )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from pinecone_text.hybrid import hybrid_convex_scale  # noqa:F401
            from pinecone_text.sparse.base_sparse_encoder import (
                BaseSparseEncoder,  # noqa:F401
            )
        except ImportError:
            raise ImportError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
            **kwargs,
        )
        final_result = []
        for res in result["matches"]:
            context = res["metadata"].pop(self.text_content_key)
            metadata = res["metadata"]
            if "score" not in metadata and "score" in res:
                metadata["score"] = res["score"]
            final_result.append(Document(page_content=context, metadata=metadata))
        # return search results as json
        return final_result
    
    def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)

        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # if isinstance(dense_vec, list):
        #     dense_vec = np.array(dense_vec)
        #     if len(dense_vec.shape) == 1:  # reshape to 2D to match the sparse shape
        #         dense_vec = dense_vec.reshape(1, -1)
        #         print(f"Dense vector shape: {dense_vec.shape}")

        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
            **kwargs,
        )
        final_result = []
        for res in result["matches"]:
            context = res["metadata"].pop(self.text_content_key)
            metadata = res["metadata"]
            if "score" not in metadata and "score" in res:
                metadata["score"] = res["score"]
            final_result.append(Document(page_content=context, metadata=metadata))
        # return search results as json
        return final_result
