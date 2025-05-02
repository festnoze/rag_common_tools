import os
import time
from typing import List, Optional, Union
import json
from collections import defaultdict

# langchain related imports
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# common tools imports
from common_tools.helpers.txt_helper import txt
from common_tools.helpers.llm_helper import Llm
from common_tools.models.llm_info import LlmInfo
from common_tools.helpers.file_helper import file
from common_tools.helpers.env_helper import EnvHelper
from common_tools.langchains.langchain_factory import LangChainFactory
from common_tools.models.embedding_model import EmbeddingModel
from common_tools.models.embedding_model_factory import EmbeddingModelFactory
from common_tools.models.vector_db_type import VectorDbType

class RagService:
    def __init__(self, llms_or_info: Optional[Union[LlmInfo, Runnable, list]], embedding_model:EmbeddingModel= None, vector_db_base_path:str = None, vector_db_type:VectorDbType = VectorDbType('chroma'), vector_db_name:str = 'main', documents_json_filename:str = None):
        # Init default parameters values if not setted
        if not vector_db_base_path: vector_db_base_path = './storage'
        if not documents_json_filename: documents_json_filename = 'bm25_documents.json'
        self.llms_infos: list[LlmInfo] = None
        self.llm_1=None
        self.llm_2=None
        self.llm_3=None
        self.instanciate_embedding(embedding_model)
        self.instanciate_llms(llms_or_info, test_llms_inference=False)
        self.vector_db_name:str = vector_db_name
        self.vector_db_type:VectorDbType = vector_db_type
        self.vector_db_base_path:str = vector_db_base_path
        self.vector_db_full_name = self.get_vectorstore_full_name(self.vector_db_name)
        self.vector_db_path:str = os.path.join(os.path.join(os.path.abspath(vector_db_base_path), self.embedding_model_name), vector_db_type.value)
        #
        self.all_documents_json_file_path = os.path.abspath(os.path.join(vector_db_base_path, documents_json_filename))
        self.vectorstore:VectorStore = self.load_vectorstore(self.vector_db_path, self.embedding, self.vector_db_type, self.vector_db_full_name)

        self.langchain_documents:list[Document] = None
        is_BM25_docs_stored_in_db = EnvHelper.get_BM25_storage_as_db_sparse_vectors()
        if not is_BM25_docs_stored_in_db:
            self.langchain_documents:list[Document] = self.load_raw_langchain_documents(self.all_documents_json_file_path)

    def instanciate_embedding(self, embedding_model:EmbeddingModel):
        self.embedding = EmbeddingModelFactory.create_instance(embedding_model)
        self.embedding_model_name = embedding_model.model_name
        
    def instanciate_llms(self, llm_or_infos: Optional[Union[LlmInfo, Runnable, list]], test_llms_inference:bool = False):        
        if isinstance(llm_or_infos, list):
            if any(llm_or_infos) and isinstance(llm_or_infos[0], LlmInfo):
                self.llms_infos = llm_or_infos
            index = 1
            for llm_or_info in llm_or_infos:
                if index == 1:
                    self.llm_1 = self.init_llm(llm_or_info, test_llms_inference)
                elif index == 2:
                    self.llm_2 = self.init_llm(llm_or_info, test_llms_inference)
                elif index == 3:
                    self.llm_3 = self.init_llm(llm_or_info, test_llms_inference)
                else:
                    raise ValueError("Only 4 llms are supported")
                index += 1
        else:
            self.llm_1 = self.init_llm(llm_or_infos)
        
        #set default llms if undefined
        if not self.llm_2: self.llm_2 = self.llm_1
        if not self.llm_3: self.llm_3 = self.llm_2
    
    def init_llm(self, llm_or_info: Optional[Union[LlmInfo, Runnable]], test_inference:bool = False) -> Runnable:
        if isinstance(llm_or_info, LlmInfo) or (isinstance(llm_or_info, list) and any(llm_or_info) and isinstance(llm_or_info[0], LlmInfo)):            
            llm = LangChainFactory.create_llms_from_infos(llm_or_info)[0]
            if test_inference:
                if not Llm.test_llm_inference(llm):                    
                    model_name = llm.model_name if hasattr(llm, 'model_name') else llm.model if hasattr(llm, 'model') else llm.__class__.__name__
                    raise ValueError(f"Inference test failed for model: '{model_name}'.")
            return llm
        elif isinstance(llm_or_info, Runnable):
            return llm_or_info
        else:
            raise ValueError("Invalid llm_or_infos parameter")
  
    def load_raw_langchain_documents(self, filepath:str = None) -> List[Document]:
        if not file.exists(filepath):
            txt.print(">>> No file found to loading langchain documents for BM25 retrieval. Please generate them first or provide a valid file path.")
            return None
                        
        json_as_str = file.read_file(filepath)
        json_data = json.loads(json_as_str)
        docs = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
            for doc in json_data 
        ]
        return docs
    
    def load_vectorstore(self, vector_db_path:str = None, embedding: Embeddings = None, vectorstore_type: VectorDbType = VectorDbType('chroma'), vector_db_full_name:str = 'default') -> VectorStore:
        try:
            is_cloud_hosted_db = (vectorstore_type == VectorDbType.Pinecone) # TODO: to generalize (example: Qdrant handle both cloud and self hosted)
            vectorstore:VectorStore = None

            if not is_cloud_hosted_db:
                vector_db_path = os.path.join(vector_db_path, vector_db_full_name)
                if not file.exists(vector_db_path): 
                    txt.print(f'>> Vectorstore not loaded, as path: "... {vector_db_path[-110:]}" is not found')
            
            if vectorstore_type == VectorDbType.ChromaDB:
                from langchain_chroma import Chroma
                #
                vectorstore = Chroma(persist_directory= vector_db_path, embedding_function= embedding)
                print(f"✓ Loaded Chroma vectorstore: '{vector_db_full_name}'.")
                
            elif vectorstore_type == VectorDbType.Qdrant:
                from qdrant_client import QdrantClient
                from langchain_qdrant import QdrantVectorStore
                #
                qdrant_client = QdrantClient(path=vector_db_path)
                vectorstore = QdrantVectorStore(client=qdrant_client, collection_name=vector_db_full_name, embedding=embedding)
                print(f"✓ Loaded Qdrant vectorstore: '{vector_db_full_name}'.")
                
            elif vectorstore_type == VectorDbType.Pinecone:
                import pinecone
                from pinecone import ServerlessSpec
                from langchain_pinecone import PineconeVectorStore
                #
                pinecone_instance = pinecone.Pinecone(api_key= EnvHelper.get_pinecone_api_key()) #, environment= EnvHelper.get_pinecone_environment()                
                
                # Create the DB (Pinecone's index) if it doesn't exist yet
                if vector_db_full_name not in pinecone_instance.list_indexes().names():
                    embedding_vector_size = len(self.embedding.embed_query("test"))   
                    is_native_hybrid_search = EnvHelper.get_is_common_db_for_sparse_and_dense_vectors()  
                    #               
                    pinecone_instance.create_index(
                                        name= vector_db_full_name, 
                                        dimension=embedding_vector_size,
                                        metric= "dotproduct" if is_native_hybrid_search else "cosine",
                                        #pod_type="s1",
                                        spec=ServerlessSpec(
                                                cloud='aws',
                                                region='us-east-1'
                                        )
                    )
                    
                    while not pinecone_instance.describe_index(vector_db_full_name).status['ready']:
                        time.sleep(1)
                    
                pinecone_index = pinecone_instance.Index(name=vector_db_full_name)
                print(f"✓ Loaded Pinecone vectorstore: '{vector_db_full_name}' index, containing " + str(pinecone_index.describe_index_stats()['total_vector_count']) + " vectors total.")
                vectorstore = PineconeVectorStore(index=pinecone_index, embedding=self.embedding)
            return vectorstore
        
        except Exception as e:
            txt.print(f"/!\\ ERROR Loading vectorstore named: '{vector_db_full_name}' /!\\: {e}")
            return None

    def get_vectorstore_full_name(self, vectorstore_base_name):
        is_native_hybrid_search = EnvHelper.get_is_common_db_for_sparse_and_dense_vectors()
        is_summarized = EnvHelper.get_is_summarized_data()
        has_questions = EnvHelper.get_is_questions_created_from_data()

        vectorstore_name_postfix = f"{'-summary' if is_summarized else '-full'}{'-quest' if has_questions else ''}{'-hybrid' if is_native_hybrid_search else ''}"
            
        # Make the index name specific regarding: native hybrid search (both sparse & dense vectors in the same record), w/ summaries, w/ questions
        vector_db_full_name = vectorstore_base_name + vectorstore_name_postfix
            
            # Limit the max length of a vectorstore name  an index name is 45 characters in pinecone
        if len(vector_db_full_name) > 45:
            txt.print(f"/!\\ Vectorstore name: '{vector_db_full_name}' is too long (Pinecone index names are 45 characters max.) and has been truncated.")
            vector_db_full_name = vector_db_full_name[:45]
        return vector_db_full_name