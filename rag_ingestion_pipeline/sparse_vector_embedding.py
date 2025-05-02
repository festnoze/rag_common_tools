import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from langchain_core.documents import Document
#
from common_tools.helpers.file_helper import file
from common_tools.helpers.txt_helper import txt

class SparseVectorEmbedding:
    vectorizer: TfidfVectorizer = None
    file_base_path:str = None
    sparse_vectorizer_filename:str = None

    def __init__(self, file_base_path, file_prefix, load_vectorizer_from_file=True, k1=1.5, b=0.75):
        if not SparseVectorEmbedding.file_base_path:
            SparseVectorEmbedding.file_base_path = file_base_path
        if not SparseVectorEmbedding.sparse_vectorizer_filename:
            SparseVectorEmbedding.sparse_vectorizer_filename = file_prefix + "_sparse_vectorizer.pkl"
            
        self.k1 = k1
        self.b = b
        self.avg_doc_length = None  # Stores the average document length after initial embedding
        
        if load_vectorizer_from_file:
            self.load_vectorizer()
        else:
            self.create_new_vectorizer()
    
    def load_vectorizer(self):
        if SparseVectorEmbedding.vectorizer: 
            return
        if not SparseVectorEmbedding.file_base_path: 
            raise ValueError("SparseVectorEmbedding.file_base_path is not set. Please set it before loading the vectorizer.")   
        filepath = os.path.join(SparseVectorEmbedding.file_base_path, SparseVectorEmbedding.sparse_vectorizer_filename)
        if not file.exists(filepath): 
            raise ValueError(f"File '{filepath}' not found in {self.load_or_create_vectorizer.__name__}.")

        # Load existing vectorizer from file
        with open(filepath, 'rb') as f:
            SparseVectorEmbedding.vectorizer = pickle.load(f)
        return

    def create_new_vectorizer(self):
        """ Create a new vectorizer (remove the existing *.pkl file if exists). """
        if SparseVectorEmbedding.vectorizer: return
        if not SparseVectorEmbedding.file_base_path: raise ValueError("SparseVectorEmbedding.file_base_path is not set. Please set it before creating the vectorizer.")
        filepath = os.path.join(SparseVectorEmbedding.file_base_path, SparseVectorEmbedding.sparse_vectorizer_filename)
        
        SparseVectorEmbedding.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, use_idf=True)
        if file.exists(filepath):
            file.delete_file(filepath)
            txt.print(f"/!\\ Previous existing Sparse Vectorizer file deleted: '{filepath}'.")

    def encode_queries(self, query:str):
        csr_matrix = self.embed_documents_as_csr_matrix_sparse_vectors_for_TF_IDF([query])
        return self.csr_to_pinecone_sparse_vector_dict(csr_matrix)
    
    def embed_documents_as_csr_matrix_sparse_vectors_for_TF_IDF(self, docs: list[str]) -> csr_matrix:
        # Vectorizing documents into sparse vectors with TfidfVectorizer
        sparse_vectors = SparseVectorEmbedding.vectorizer.fit_transform(docs)  # Sparse matrix
        txt.print(f"Shape of the sparse vectors: {sparse_vectors.shape}")
        vocabulary_size = len(self.vectorizer.get_feature_names_out())  # Total unique tokens
        txt.print(f"Size of the sparse vectors: {vocabulary_size}")

        # Conversion of TF-IDF vectors to BM25-compatible format (normalizes and adjusts weights)
        bm25_vectors = csr_matrix(sparse_vectors)  # Sparse matrix
        return bm25_vectors
    
    def embed_documents_as_sparse_vectors_for_BM25_initial(self, documents_contents:list[str]):
        """
        Embeds a set of documents as BM25-compatible sparse vectors. 
        Learns vocabulary, IDF, and document statistics during this process.
        """
        # Learn vocabulary and IDF, calculate sparse TF matrix
        tf = SparseVectorEmbedding.vectorizer.fit_transform(documents_contents)  # Sparse term-frequency matrix

        # Compute average document length
        doc_lengths = np.array(tf.sum(axis=1)).flatten()  # Row-wise sum (document lengths)
        self.avg_doc_length = np.mean(doc_lengths)

        # Compute IDF values
        idf = SparseVectorEmbedding.vectorizer.idf_

        # Apply BM25 weighting
        rows, cols = tf.nonzero()
        bm25_data = []
        for row, col in zip(rows, cols):
            tf_value = tf[row, col]
            doc_length = doc_lengths[row]
            idf_value = idf[col]
            denominator = tf_value + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            bm25_value = idf_value * ((tf_value * (self.k1 + 1)) / denominator)
            bm25_data.append(bm25_value)

        # Construct sparse BM25 matrix
        bm25 = csr_matrix((bm25_data, (rows, cols)), shape=tf.shape)
        return bm25

    def embed_documents_as_sparse_vectors_for_BM25_upon_previous_vocabulary(self, new_documents_contents:list[str]):
        """
        Embeds a new set of documents as BM25-compatible sparse vectors 
        using the vocabulary and statistics learned from the initial set of documents.
        """
        if self.avg_doc_length is None:
            raise ValueError("The vectorizer needs to embed the initial documents first (call embed_documents_as_sparse_vectors_for_BM25_initial).")
        
        # Transform using the existing vocabulary
        tf = SparseVectorEmbedding.vectorizer.transform(new_documents_contents)  # Sparse term-frequency matrix
        doc_lengths = np.array(tf.sum(axis=1)).flatten()  # Row-wise sum (document lengths)

        # Apply BM25 weighting
        rows, cols = tf.nonzero()
        bm25_data = []
        for row, col in zip(rows, cols):
            tf_value = tf[row, col]
            doc_length = doc_lengths[row]
            idf_value = self.vectorizer.idf_[col]
            denominator = tf_value + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            bm25_value = idf_value * ((tf_value * (self.k1 + 1)) / denominator)
            bm25_data.append(bm25_value)

        # Construct sparse BM25 matrix
        bm25 = csr_matrix((bm25_data, (rows, cols)), shape=tf.shape)
        return bm25
    
    def csr_to_pinecone_sparse_vector_dict(self, csr_matrix):
        """Convert a CSR sparse matrix into Pinecone-compatible sparse_values format."""
        coo = csr_matrix.tocoo()  # Convert to COO format
        return {
            "indices": coo.col.tolist(),  # Indices of non-zero values
            "values": coo.data.tolist()   # Values of non-zero entries
        }
    
    def save_vectorizer(self):
        if not SparseVectorEmbedding.file_base_path: 
            raise ValueError("SparseVectorEmbedding.file_base_path is not set. Please set it before saving the vectorizer.")
        filepath = os.path.join(SparseVectorEmbedding.file_base_path, SparseVectorEmbedding.sparse_vectorizer_filename)
        with open(filepath, 'wb') as f:
            pickle.dump(SparseVectorEmbedding.vectorizer, f)