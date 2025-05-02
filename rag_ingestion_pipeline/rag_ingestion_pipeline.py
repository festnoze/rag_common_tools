from collections import defaultdict
import os
import re
import json
import sys
import time
from typing import Union
import uuid

# langchain related imports
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
import numpy as np

# common tools imports
from common_tools.helpers.txt_helper import txt
from common_tools.models.file_already_exists_policy import FileAlreadyExistsPolicy
from common_tools.helpers.file_helper import file
from rag_common_tools.rag_ingestion_pipeline.sparse_vector_embedding import SparseVectorEmbedding
from rag_common_tools.rag_service import RagService
from common_tools.models.vector_db_type import VectorDbType
from common_tools.helpers.batch_helper import BatchHelper
from rag_common_tools.rag_ingestion_pipeline.rag_chunking import RagChunking
from common_tools.helpers.env_helper import EnvHelper

class RagIngestionPipeline:
    def __init__(self, rag: RagService):
        self.rag_service: RagService = rag

    def chunk_documents(self, documents: list) -> list[Document]:
        """Chunks the provided documents into small pieces"""
        documents_chunks:list[Document] = []
        embedding_size = EnvHelper.get_embedding_size()
        if embedding_size != 0:
            txt.print_with_spinner("Splitting documents into chunks...")
            documents_chunks = RagChunking.split_text_into_chunks(documents, embedding_size, embedding_size/10)  

            # Set new metadata id for each document which has been chunked
            RagIngestionPipeline._set_new_unique_id_to_chunked_docs(documents_chunks)          
            txt.stop_spinner_replace_text("Documents successfully split into chunks in " + str(len(documents_chunks)) + " chunks.")      
        
        else:
            if isinstance(documents[0], Document):
                documents_chunks = documents
            else:
                documents_chunks = [
                    Document(page_content=doc["page_content"], 
                            metadata=doc["metadata"] if doc["metadata"] else '') 
                    for doc in documents
                ]
        return documents_chunks

    def _set_new_unique_id_to_chunked_docs(documents_chunks: list[Document]) -> None:
        doc_id_to_docs = defaultdict(list)
        for doc in documents_chunks:
            doc_id = doc.metadata.get('doc_id')
            if doc_id is not None:
                doc_id_to_docs[doc_id].append(doc)

        for docs in doc_id_to_docs.values():
            if len(docs) > 1:
                for doc in docs:
                    doc.metadata['id'] = str(uuid.uuid4())

    def embed_chunks_then_add_to_vectorstore(self, docs_chunks: list, vector_db_type: VectorDbType, collection_name:str, delete_existing= True, load_embeddings_from_file_if_exists= True) -> any:
        """
        Add to the vector store provided chunked documents, after embedding them.
        Args:embed_chunked_docs_then_add_to_vectorstore
            chunks (list): List of document chunks to be embedded and stored.
            vector_db_type (VectorDbType, optional): Type of vector database to use. Defaults to 'chroma'.
            collection_name (str, optional): Name of the collection in the vector database. Defaults to 'main'.
            delete_existing (bool, optional): Flag to determine if existing vector store should be deleted before building a new one. Defaults to True.
        Returns:
            any: The database object after storing the document chunks.
        """
        if not vector_db_type: vector_db_type = VectorDbType('chroma')
        if not docs_chunks or len(docs_chunks) == 0: raise ValueError("No documents provided")
        if not hasattr(self.rag_service, 'embedding') or not self.rag_service.embedding: raise ValueError("Embedding model must be specified to build vector store")
        if delete_existing:
            self._reset_vectorstore(self.rag_service)        
        BM25_storage_in_db_as_sparse_vectors = EnvHelper.get_BM25_storage_as_db_sparse_vectors()
        
        #  Store embedded chunks as dense vectors into vector database
        #  + Create a Json file containing the raw chunks for BM25 retrieval
        if not BM25_storage_in_db_as_sparse_vectors:
            self._build_bm25_store_as_raw_json_file(docs_chunks)
            
            if vector_db_type == VectorDbType.Qdrant:
                db = self._embed_and_store_documents_chunks_as_dense_vectors_into_qdrant_db(chunks= docs_chunks, embedding = self.rag_service.embedding, vector_db_path= self.rag_service.vector_db_path, collection_name=collection_name)
            elif vector_db_type == VectorDbType.ChromaDB:
                db = self._embed_and_store_documents_chunks_as_dense_vectors_into_chroma_db(chunks= docs_chunks, embedding = self.rag_service.embedding, vector_db_path= self.rag_service.vector_db_path, collection_name=collection_name)
            elif vector_db_type == VectorDbType.Pinecone:
                db = self._embed_and_store_documents_chunks_as_dense_vectors_into_pinecone_db(chunks= docs_chunks)
            else:
                raise ValueError("Invalid vector db type: " + vector_db_type.value)

        # Store embedded chunks as dense + sparse vectors into vector database
        # Allow performing both semantic and BM25 retrieval from vector database
        # TODO: in the same DB for now, but could be in separates databases in the future, using "IS_COMMON_DB_FOR_SPARSE_AND_DENSE_VECTORS" config key with value = False
        elif BM25_storage_in_db_as_sparse_vectors:
            if vector_db_type == VectorDbType.Qdrant:
                raise NotImplementedError("Sparse vectors for BM25 are not implemented for Qdrant.")
            elif vector_db_type == VectorDbType.ChromaDB:
                raise NotImplementedError("Sparse vectors for BM25 are not implemented for Chroma.")
            elif vector_db_type == VectorDbType.Pinecone:
                db = self._embed_documents_as_dense_and_sparse_vectors_and_store_into_pinecone_db(
                                chunks= docs_chunks,
                                pinecone_index= self.rag_service.vectorstore._index,
                                embedding_model= self.rag_service.embedding,
                                load_embeddings_from_file_if_exists= load_embeddings_from_file_if_exists)
        txt.stop_spinner_replace_text(f"Done. {len(docs_chunks)} documents' chunks embedded sucessfully!")
        return db

    def _embed_and_store_documents_chunks_as_dense_vectors_into_chroma_db(self, chunks:list[Document], embedding, vector_db_path:str, collection_name:str = 'main', batch_size:int = 2000):
        total_elapsed_seconds = 0
        batchs = BatchHelper.batch_split_by_count(chunks, batch_size)
        for i, batch in enumerate(batchs):
            txt.print_with_spinner(f"Batch n°{i+1}/{batchs}: Embedding {len(batch)} documents ")
            db = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory= os.path.join(vector_db_path, collection_name)
            )
            total_elapsed_seconds += txt.stop_spinner_replace_text(f"Batch n°{i+1}/{batchs} done. {len(chunks)} chunks embedded sucessfully.") 
        
        txt.print(f"All documents sucessfully uploaded into Chroma database in: {total_elapsed_seconds}s.")         
        return db
    
    def _embed_and_store_documents_chunks_as_dense_vectors_into_qdrant_db(self, chunks:list[Document], embedding:Embeddings, vector_db_path: str = '', collection_name:str = 'main', batch_size:int = 2000) -> QdrantVectorStore:
        txt.print(f"Start embedding of {len(chunks)} chunks of documents...")
        vector_size = len(embedding.embed_query("test"))  # Determine the vector size
        qdrant_client = QdrantClient(path=vector_db_path)
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        db = QdrantVectorStore(
                client= qdrant_client,
                collection_name= collection_name,
                embedding= embedding)
        total_elapsed_seconds = 0
        batchs = BatchHelper.batch_split_by_count(chunks, batch_size)
        for i, batch in enumerate(batchs):
            txt.print_with_spinner(f"Batch n°{i+1}/{batchs}: Embedding {len(batch)} documents ")
            db.add_documents(batch)
            total_elapsed_seconds += txt.stop_spinner_replace_text(f"Batch n°{i+1}/{batchs} done. {len(chunks)} chunks embedded sucessfully.")
        
        txt.print(f"All documents sucessfully uploaded into Qdrant database in: {total_elapsed_seconds}s.")
        return db
     
    def _embed_and_store_documents_chunks_as_dense_vectors_into_pinecone_db(self, chunks:list[Document], batch_size:int = 2000) -> PineconeVectorStore:
        total_elapsed_seconds = 0
        batchs = BatchHelper.batch_split_by_count(chunks, batch_size)
        for i, batch in enumerate(batchs):
            txt.print_with_spinner(f"Batch n°{i+1}/{batchs}: Embedding {len(batch)} documents ")
            self.rag_service.vectorstore.add_documents(batch)
            total_elapsed_seconds += txt.stop_spinner_replace_text(f"Batch n°{i+1}/{batchs} done. {len(chunks)} chunks embedded sucessfully.")
        
        txt.print(f"All documents sucessfully uploaded into Pinecone database in: {total_elapsed_seconds}s.")
        return self.rag_service.vectorstore
    
    def _embed_documents_as_dense_and_sparse_vectors_and_store_into_pinecone_db(self, chunks: list[Document], pinecone_index, embedding_model: Embeddings, load_embeddings_from_file_if_exists = True, batch_size_in_mega_bytes: int = 1):
        """
        Embeds documents with BM25 (sparse) and dense embeddings and stores them in Pinecone.
        
        Args:
            documents (list[Document]): List of LangChain Document objects.
            pinecone_index: Pinecone index instance.
            bm25_embedding_model: Model or method to compute BM25 sparse vectors.
            dense_embedding_model: Model or method to compute dense embeddings.
        """
        # Create (or load) both sparse and dense embeddings for all documents
        all_entries = self.load_or_compute_sparse_and_dense_vectors_embeddings_for_chunks(
                                chunks= chunks,
                                embedding_model= embedding_model,
                                load_embeddings_from_file_if_exists= load_embeddings_from_file_if_exists,
                                batch_embedding_size= 1000,
                                wait_seconds_btw_batches= None)

        # Insert the sparse + dense embeddings into the current Pinecone vector database (index)
        total_elapsed_seconds = 0
        insertion_batches = BatchHelper.batch_split_by_size_in_kilo_bytes(all_entries, batch_size_in_mega_bytes * 1024)
        for i, batch_entries in enumerate(insertion_batches):
            txt.print_with_spinner(f"Inserting {len(batch_entries)} documents into Pinecone DB. Batch n°{i+1}/{len(insertion_batches)}.")
            pinecone_index.upsert(batch_entries)
            total_elapsed_seconds += txt.stop_spinner_replace_text(f"Batch n°{i+1}/{len(insertion_batches)} done. {len(batch_entries)} documents' chunks inserted sucessfully into database.")

        txt.print(f"All documents sucessfully uploaded into Pinecone database in: {total_elapsed_seconds}s.")
        return Pinecone(index=pinecone_index, embedding=embedding_model)

    def load_or_compute_sparse_and_dense_vectors_embeddings_for_chunks(self, chunks:list[Document], embedding_model:Embeddings, load_embeddings_from_file_if_exists:bool = True, batch_embedding_size:int = 1000, wait_seconds_btw_batches:float = None) -> list[dict]:
        # Try to load existing embeddings (spase + dense) from file
        joined_embeddings_filepath = os.path.join(self.rag_service.vector_db_base_path, self.rag_service.vector_db_full_name + "_joined_sparse_and_dense_embeddings_for_pinecone.json")
        if load_embeddings_from_file_if_exists and file.exists(joined_embeddings_filepath):
            all_entries = file.get_as_json(joined_embeddings_filepath)
            print(f"!!! Loaded {len(all_entries)} existing entries (sparse + dense vectors) from file: {joined_embeddings_filepath} !!!")
            self.fix_empty_sparse_vectors(all_entries)
            return all_entries
        
        # Embed all documents as sparse vectors and dense vectors 
        # (as no file containing previous embeddings exists, or if the user ask to recompute the embeddings)
        sparse_vectorizer = SparseVectorEmbedding(self.rag_service.vector_db_base_path, self.rag_service.vector_db_full_name, load_vectorizer_from_file= False)
        all_chunks_contents = [doc.page_content for doc in chunks]
            
        # Step 1: Compute Sparse Vectors (BM25)
        txt.print_with_spinner(f"Sparse embedding of {len(all_chunks_contents)} chunks as BM25 in progress ...")
        sparse_vectors = sparse_vectorizer.embed_documents_as_sparse_vectors_for_BM25_initial(all_chunks_contents)
        sparse_vectorizer.save_vectorizer()
        txt.stop_spinner_replace_text(f"BM25 sparse vectors embedded sucessfully. Sparse vectorizer saved in: {sparse_vectorizer.file_base_path}.")

        # Step 2: Compute or Load Dense Vectors
        dense_vectors_filename = self.rag_service.vector_db_full_name + "_dense_vectors.npy"
        dense_vectors_filepath = os.path.join(self.rag_service.vector_db_base_path, dense_vectors_filename)
        dense_vectors = []
        if load_embeddings_from_file_if_exists and file.exists(dense_vectors_filepath):
            dense_vectors_array = np.load(dense_vectors_filepath)
            dense_vectors = dense_vectors_array.tolist()
            txt.print(f"!!! Loaded {len(dense_vectors)} existing dense vectors from file: {dense_vectors_filepath} !!!")
        else:
            total_elapsed_seconds = 0
            batchs = BatchHelper.batch_split_by_count(all_chunks_contents, batch_embedding_size)
            txt.print(f">>> Start embedding as dense vectors: {len(all_chunks_contents)} documents, splitted into {len(batchs)} batches.")
            for i, batch_chunks_contents in enumerate(batchs):
                txt.print_with_spinner(f"Embedding dense vectors: Batch n°{i+1}/{len(batchs)}: {len(batch_chunks_contents)} documents")
                batch_dense_vectors = embedding_model.embed_documents(batch_chunks_contents)
                dense_vectors.extend(batch_dense_vectors)
                if wait_seconds_btw_batches: 
                    time.sleep(wait_seconds_btw_batches)
                total_elapsed_seconds += txt.stop_spinner_replace_text(f"Batch n°{i+1}/{len(batchs)} done. {len(batch_dense_vectors)}/{len(chunks)} dense vectors embedded sucessfully.")
            dense_vectors_array = np.array(dense_vectors)
            np.save(dense_vectors_filepath, dense_vectors_array)
            txt.print(f"All dense + sparse vectors sucessfully embedded in: {total_elapsed_seconds:2f}s.")

        # Step 3: Prepare joined sparse and dense embedding dict (compatible with Pinecone Entries) - also inc. metadata from the original documents
        all_entries = []
        all_entries_ids = []
        for doc, sparse_vector, dense_vector in zip(chunks, sparse_vectors, dense_vectors):            
            bm25_sparse_dict = sparse_vectorizer.csr_to_pinecone_sparse_vector_dict(sparse_vector) # Convert CSR matrix to Pinecone dictionary
            doc.metadata["text"] = doc.page_content  # Add the corresponding text content into metadata
            entry_id = doc.metadata.get("id", str(uuid.uuid4()))
            # Make sure that the entry id is unique
            while entry_id in all_entries_ids:
                print(f"!!! /!\\ Duplicate entry 'id' detected: '{entry_id}'. New id generated !!!")
                entry_id = str(uuid.uuid4())

            if doc.metadata["id"] != entry_id:  
                print(f"!!! /!\\ Entry 'id' changed from '{doc.metadata['id']}' to '{entry_id}' !!!")
                doc.metadata["id"] = entry_id
            all_entries_ids.append(entry_id)
                
            # Combine sparse and dense vectors as two fields of a single entry (correspond to Pinecone's structure)
            entry = {
                    "id": entry_id,  # Add a unique id for each entry
                    "values": dense_vector,  # Pinecone handles dense vectors in the 'values' field
                    "sparse_values": bm25_sparse_dict,  # BM25 sparse vector for hybrid search
                    "metadata": doc.metadata  # Add metadata for filtering
                }
            all_entries.append(entry)
        self.fix_empty_sparse_vectors(all_entries)
        
        # Save joined embeddings as file 
        all_entries_json = json.dumps(all_entries, ensure_ascii=False, indent=4)
        file.write_file(all_entries_json, joined_embeddings_filepath, FileAlreadyExistsPolicy.Override)
        return all_entries

    def fix_empty_sparse_vectors(self, all_entries):
        for entry in all_entries:
            if len(entry["sparse_values"]["indices"]) == 0:
                entry["sparse_values"]["indices"] = all_entries[0]["sparse_values"]["indices"]
                print(f"!!! /!\\ Sparse vector 'indices' are empty. Using the first entry's indices: {entry['sparse_values']['indices']} !!!")
            if len(entry["sparse_values"]["values"]) == 0:
                entry["sparse_values"]["values"] = all_entries[0]["sparse_values"]["values"]
                print(f"!!! /!\\ Sparse vector 'values' are empty. Using the first entry's values: {entry['sparse_values']['values']} !!!")

    def _build_bm25_store_as_raw_json_file(self, documents:list):
        documents_dict = []
        for document in documents:
            if isinstance(document, Document):
                documents_dict.append(self._build_document(document.page_content, document.metadata))
            elif isinstance(document, dict):
                documents_dict.append(self._build_document(document["page_content"], document["metadata"]))
            else:
                raise ValueError("Invalid data type")
        
        json_data = json.dumps(documents_dict, ensure_ascii=False, indent=4)
        file.write_file(json_data, self.rag_service.all_documents_json_file_path, file_exists_policy= FileAlreadyExistsPolicy.Override)

    def _reset_vectorstore(self, rag: RagService = None):
        if rag.vectorstore:
            if rag.vector_db_type != VectorDbType.Pinecone:
                rag.vectorstore.reset_collection()
            elif rag.vector_db_type == VectorDbType.Pinecone:
                try:
                    if rag.vectorstore and rag.vectorstore._index:
                        rag.vectorstore._index.delete(delete_all=True)
                        Pinecone.delete_index(rag.vector_db_name)
                except Exception as e:
                    txt.print(f"Deleting pinecone index '{rag.vectorstore._index._config.host}' vectors fails with: {e}")

    def _get_docs_min_words_count(self, documents: list[Document]) -> int:
        return min(len(re.split(r'[ .,;:!?]', doc.page_content)) for doc in documents)
    
    def _get_docs_max_words_count(self, documents: list[Document]) -> int:
        return max(len(re.split(r'[ .,;:!?]', doc.page_content)) for doc in documents)

    def _delete_vectorstore_files(self):
        file.delete_folder(self.rag_service.vector_db_path)
    
    def _build_document(self, content: str, metadata: dict):
        return {'page_content': content, 'metadata': metadata}