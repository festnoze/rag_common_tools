from typing import Optional, Union

# langchain related imports
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pinecone_text.sparse import BM25Encoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.structured_query import (
        Comparator,
        Comparison,
        Operation,
        Operator,
        StructuredQuery,
        Visitor,
)

# internal common tools imports
from common_tools.helpers.execute_helper import Execute
from common_tools.helpers.rag_filtering_metadata_helper import RagFilteringMetadataHelper
from common_tools.helpers.env_helper import EnvHelper
from common_tools.models.conversation import Conversation
from common_tools.models.doc_w_summary_chunks_questions import DocWithSummaryChunksAndQuestions
from common_tools.models.question_analysis_base import QuestionAnalysisBase
from common_tools.models.vector_db_type import VectorDbType
from rag_common_tools.rag_service import RagService
from rag_common_tools.rag_ingestion_pipeline.sparse_vector_embedding import SparseVectorEmbedding

#from langchain_community.retrievers import PineconeHybridSearchRetriever
# ... is replaced by our own because the actual langchain code doesn't implement async _aget_relevant_documents method
from rag_common_tools.rag_inference_pipeline.custom_pinecone_hybrid_retriever import PineconeHybridSearchRetriever

class RagRetrieval:
    @staticmethod    
    async def hybrid_retrieval_langchain_async(rag: RagService, analysed_query: QuestionAnalysisBase, metadata_filters: Optional[Union[Operation, Comparison]], include_bm25_retrieval: bool = True, include_contextual_compression: bool = False, include_semantic_retrieval: bool = True, give_score: bool = True, max_retrived_count: int = 20, bm25_ratio: float = 0.2):
        if metadata_filters and not any(metadata_filters): metadata_filters = None
        do_hybrid_retrieval = include_bm25_retrieval and include_semantic_retrieval
        is_pinecone_hybrid_db = do_hybrid_retrieval and rag.vector_db_type == VectorDbType.Pinecone and EnvHelper.get_BM25_storage_as_db_sparse_vectors() and EnvHelper.get_is_common_db_for_sparse_and_dense_vectors()
        
        # Build Retriever
        if is_pinecone_hybrid_db:
            metadata_filters_in_pinecone_format = RagFilteringMetadataHelper.translate_langchain_metadata_filters_into_specified_db_type_format(metadata_filters, rag.vector_db_type)
            semantic_k_ratio = round(1 - bm25_ratio, 2)
            retriever = await RagRetrieval.pinecone_hybrid_retrieval_langchain_async(rag, max_retrived_count, semantic_k_ratio)
        else:
            retriever = RagRetrieval.get_hybrid_retriever_langchain(rag, metadata_filters, include_bm25_retrieval, include_contextual_compression, include_semantic_retrieval, give_score, max_retrived_count, bm25_ratio)
        
        # Retrieve documents
        if is_pinecone_hybrid_db:
            try:
                retrieved_chunks = retriever.invoke(QuestionAnalysisBase.get_modified_question(analysed_query), filter= metadata_filters_in_pinecone_format)
            except Exception as e:
                print(f"Error in Pinecone Hybrid Retrieval: {e}")
                retrieved_chunks = []            
        else:        
            retrieved_chunks = await retriever.ainvoke(QuestionAnalysisBase.get_modified_question(analysed_query))
            
        # In case no docs are retrieved, re-launch the hybrid retrieval, without metadata filters
        if metadata_filters and (not retrieved_chunks or not any(retrieved_chunks)):
            print('/!\\ >> Hybid retrieval returns no documents. Retrying without any metadata filters.')
            return await RagRetrieval.hybrid_retrieval_langchain_async(rag, analysed_query, None, include_bm25_retrieval, include_contextual_compression, include_semantic_retrieval, give_score, max_retrived_count, bm25_ratio)

        post_treat_retrieved_chunks = RagRetrieval.retrieval_post_treatment(rag, retrieved_chunks)
        return post_treat_retrieved_chunks

    @staticmethod
    def retrieval_post_treatment(rag: RagService, retrieved_chunks: list[Document]): 
        # Remove duplicate chunks TODO: shouldn't happends, analyse why (duplicates also happens on pinecone w/o hybrid retriever returns twice dense and sparse vectors)
        unique_chunks = []
        seen_ids = set()
        for chunk in retrieved_chunks:
            if chunk.metadata['id'] not in seen_ids:
                seen_ids.add(chunk.metadata['id'])
                unique_chunks.append(chunk)

        if len(unique_chunks) < len(retrieved_chunks):
            print(f"/!\\ {len(retrieved_chunks) - len(unique_chunks)} retrieved chunks are duplicate, and has been removed.")
        retrieved_chunks = unique_chunks

        # Remove related ids from retrieved chunks' metadata if any
        # (Those ids can be used for RAG graph, but are useless later, as for augmented generation. Limit token usage).
        if any(retrieved_chunks) and "rel_ids" in retrieved_chunks[0].metadata: 
            for retrieved_chunk in retrieved_chunks:
                if "rel_ids" in retrieved_chunk.metadata:
                    retrieved_chunk.metadata.pop("rel_ids", None)

        # Remove the questions from the text of retrieved chunks (if both questions and content are packed together in embeddings)
        if RagRetrieval._is_questions_and_content_packed_together():
            for retrieved_chunk in retrieved_chunks:
                if DocWithSummaryChunksAndQuestions.questions_title in retrieved_chunk.page_content and DocWithSummaryChunksAndQuestions.answers_title in retrieved_chunk.page_content:
                    retrieved_chunk.page_content = retrieved_chunk.page_content.split(DocWithSummaryChunksAndQuestions.answers_title, 1)[1]

        return retrieved_chunks    
    
    is_questions_and_content_packed_together: bool = None
    @staticmethod
    def _is_questions_and_content_packed_together() -> bool:
        if RagRetrieval.is_questions_and_content_packed_together is None:
            RagRetrieval.is_questions_and_content_packed_together = EnvHelper.get_is_questions_created_from_data(False) and EnvHelper.get_is_mixed_questions_and_data(False)
        return RagRetrieval.is_questions_and_content_packed_together
    
    @staticmethod    
    def get_hybrid_retriever_langchain(rag: RagService, metadata_filters: Optional[Union[Operation, Comparison]], include_bm25_retrieval: bool = True, include_contextual_compression: bool = False, include_semantic_retrieval: bool = True, give_score: bool = True, max_retrived_count: int = 20, bm25_ratio: float = 0.2):
        do_hybrid_retrieval = include_bm25_retrieval and include_semantic_retrieval
        if do_hybrid_retrieval:
            semantic_k_ratio = round(1 - bm25_ratio, 2)
        else:
            semantic_k_ratio = 1 if include_semantic_retrieval else 0
            bm25_ratio = 1 if include_bm25_retrieval else 0

        retrievers = []
        if include_semantic_retrieval:
            metadata_filters_in_specific_db_type_format = RagFilteringMetadataHelper.translate_langchain_metadata_filters_into_specified_db_type_format(metadata_filters, rag.vector_db_type)
            vector_retriever = rag.vectorstore.as_retriever(search_kwargs={'k': int(max_retrived_count * semantic_k_ratio), 'filter': metadata_filters_in_specific_db_type_format}) 
            retrievers.append(vector_retriever)
        
        if include_bm25_retrieval:
            if EnvHelper.get_BM25_storage_as_db_sparse_vectors():
                raise NotImplementedError("BM25 storage as db sparse vectors is not yet implemented for langchain retrieval.")
            else:
                # Build BM25 retriever on raw documents filtered by metadata
                if metadata_filters:
                    if RagFilteringMetadataHelper.does_contain_filter(metadata_filters, 'domaine','url'): #TODO: extract: domain specific 
                        max_retrived_count = 100
                    metadata_filters_chroma = RagFilteringMetadataHelper.translate_langchain_metadata_filters_into_chroma_db_format(metadata_filters)
                    filtered_docs = [
                        doc for doc in rag.langchain_documents 
                        if RagFilteringMetadataHelper.metadata_filtering_predicate_ChromaDb(doc, metadata_filters_chroma)
                    ]
                    print(f">> docs count corresponding to metadata: {len(filtered_docs)}/{len(rag.langchain_documents)}")
                else:
                    filtered_docs = rag.langchain_documents
                bm25_retriever = RagRetrieval.build_bm25_retriever(filtered_docs, k = int(max_retrived_count * bm25_ratio))
                retrievers.append(bm25_retriever)

        if not any(retrievers): 
            raise ValueError(f"No retriever has been defined in '{RagRetrieval.hybrid_retrieval_langchain_async.__name__}'.")

        weights = []
        if include_semantic_retrieval: weights.append(semantic_k_ratio)
        if include_bm25_retrieval: weights.append(bm25_ratio)
        ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)

        if include_contextual_compression: # move to a separate workflow step?
            _filter = LLMChainFilter.from_llm(rag.llm_1)
            final_retriever = ContextualCompressionRetriever(
                                    name= 'contextual compression retriever', 
                                    base_compressor=_filter, 
                                    base_retriever=ensemble_retriever)
        else:
            final_retriever = ensemble_retriever

        return final_retriever
    
    @staticmethod    
    async def pinecone_hybrid_retrieval_langchain_async(rag: RagService, max_retrived_count: int = 20, semantic_k_ratio: float = 0.2):
        retriever = PineconeHybridSearchRetriever(
            embeddings= rag.embedding,
            sparse_encoder= SparseVectorEmbedding(rag.vector_db_base_path, rag.vector_db_full_name, load_vectorizer_from_file= True), #TODO: use our custom sparse encoder, think to extend
            index= rag.vectorstore._index,
            top_k= max_retrived_count,  # Number of documents to retrieve
            alpha= semantic_k_ratio,  # Balance between dense and sparse vector retrieval
            namespace= rag.vectorstore._namespace,
            text_content_key= "text",  # Key in the metadata that contains the text
        )
        return retriever
        
    @staticmethod    
    def rag_hybrid_retrieval_custom(rag: RagService, analysed_query: QuestionAnalysisBase, metadata: dict, include_bm25_retrieval: bool = False, give_score: bool = True, max_retrived_count: int = 20):
        if not include_bm25_retrieval:
            rag_retrieved_chunks = RagRetrieval.semantic_vector_retrieval(rag, QuestionAnalysisBase.get_modified_question(analysed_query), metadata, give_score, max_retrived_count)
            return rag_retrieved_chunks
        
        rag_retrieved_chunks, bm25_retrieved_chunks = Execute.run_sync_functions_in_parallel_threads(
            (RagRetrieval.semantic_vector_retrieval, (rag, QuestionAnalysisBase.get_modified_question(analysed_query), metadata, give_score, max_retrived_count)),
            (RagRetrieval.bm25_retrieval, (rag, QuestionAnalysisBase.get_modified_question(analysed_query), metadata, give_score, max_retrived_count)),
        )
        retained_chunks = RagRetrieval.hybrid_chunks_selection(rag_retrieved_chunks, bm25_retrieved_chunks, give_score, max_retrived_count)
        return retained_chunks

    @staticmethod    
    def semantic_vector_retrieval(rag: RagService, query:Union[str, Conversation], metadata_filters:dict, give_score: bool = False, max_retrieved_count: int = 10, min_score: float = None, min_retrived_count: int = None):
        question_w_history = Conversation.conversation_history_as_str(query)
        if give_score:
            metadata_filter = metadata_filters if metadata_filters else None
            retrieved_chunks = rag.vectorstore.similarity_search_with_score(question_w_history, k=max_retrieved_count, filter=metadata_filter)
            if min_score and min_retrived_count and len(retrieved_chunks) > min_retrived_count:
                top_results = []
                for retrieved_chunk in retrieved_chunks:
                    if isinstance(retrieved_chunk, tuple) and retrieved_chunk[1] >= min_score or len(top_results) < min_retrived_count:
                        top_results.append(retrieved_chunk)
                retrieved_chunks = top_results
        else:
            retrieved_chunks = rag.vectorstore.similarity_search(question_w_history, k=max_retrieved_count, filter=metadata_filters)
        return retrieved_chunks

    @staticmethod
    def build_bm25_retriever(documents: list[Document], k: int = 20, metadata: dict = None, action_name = 'RAG BM25 retrieval') -> BM25Retriever:
        if not documents or len(documents) == 0: 
            return None
        
        if metadata:
            bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in documents], metadata)
        else:
            bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        return bm25_retriever.with_config({"run_name": f"{action_name}"})
    
    @staticmethod
    def bm25_retrieval(rag: RagService, query:Union[str, Conversation], metadata_filters: dict, give_score: bool, k = 3):
        if metadata_filters and any(metadata_filters):
            filtered_docs = [doc for doc in rag.langchain_documents if RagFilteringMetadataHelper.metadata_filtering_predicate_ChromaDb(doc, metadata_filters)]
        else:
            filtered_docs = rag.langchain_documents

        question_w_history = Conversation.conversation_history_as_str(query)
        bm25_retriever = RagRetrieval.build_bm25_retriever(filtered_docs, k)
        bm25_retrieved_chunks = bm25_retriever.invoke(question_w_history)
       
        if give_score:
            score = 0.1 #todo: define the score
            return [(doc, score) for doc in bm25_retrieved_chunks]
        else:
            return bm25_retrieved_chunks
        
    @staticmethod   
    def hybrid_chunks_selection(rag_retrieved_chunks: list[tuple[Document, float]], bm25_retrieved_chunks: list[tuple[Document, float]] = None, give_score: bool = False, max_retrived_count: int = None):
        """ Chunks ranking and selection from both semantic and bm25 retrievals.
        """
        if not bm25_retrieved_chunks or not any(bm25_retrieved_chunks):
            return rag_retrieved_chunks
        
        rag_retrieved_chunks.extend([(chunk, 0) for chunk in bm25_retrieved_chunks] if give_score else bm25_retrieved_chunks)
        
        if max_retrived_count:
            if give_score:
                rag_retrieved_chunks = sorted(rag_retrieved_chunks, key=lambda x: x[1], reverse=True)
            rag_retrieved_chunks = rag_retrieved_chunks[:max_retrived_count]

        return rag_retrieved_chunks
    
    @staticmethod   
    def chunks_reranking_and_selection(retrieved_chunks: list[tuple[Document, float]]):
        return retrieved_chunks
    