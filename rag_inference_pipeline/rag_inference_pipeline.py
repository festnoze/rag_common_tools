import asyncio
from typing import Union, Generator
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import Tool
#from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_experimental.utilities import PythonREPL

# Import the task classes from other files
from common_tools.helpers.llm_helper import Llm
from common_tools.helpers.execute_helper import Execute
from common_tools.helpers.rag_filtering_metadata_helper import RagFilteringMetadataHelper
from rag_common_tools.rag_service import RagService
from rag_common_tools.rag_inference_pipeline.rag_pre_treatment_tasks import RAGPreTreatment
from rag_common_tools.rag_inference_pipeline.rag_guardrails_tasks import RAGGuardrails
from rag_common_tools.rag_inference_pipeline.rag_retrieval import RagRetrieval
from rag_common_tools.rag_inference_pipeline.rag_augmented_generation_tasks import RAGAugmentedGeneration
from rag_common_tools.rag_inference_pipeline.rag_post_treatment_tasks import RAGPostTreatment
from common_tools.helpers.ressource_helper import Ressource
from common_tools.workflows.workflow_executor import WorkflowExecutor
from common_tools.models.conversation import Conversation
from common_tools.workflows.end_workflow_exception import EndWorkflowException
from common_tools.helpers.env_helper import EnvHelper

class RagInferencePipeline:
    def __init__(self, rag:RagService, override_workflow_available_classes:dict = None, default_filters:dict = {}, metadata_descriptions = None, tools: list = None):
        self.rag:RagService = rag
        self.workflow_concrete_classes:dict = None
        self.set_workflow_concrete_classes(override_workflow_available_classes)
        self.default_filters = default_filters

        # If the metadata descriptions are not provided upon instanciation...
        # generate them from the metadata with their values found within all the documents
        if not metadata_descriptions:
            if not rag.langchain_documents:
                if EnvHelper.get_BM25_storage_as_db_sparse_vectors():
                    raise ValueError("The metadata descriptions cannot be generated because langchain_documents isn't set in RAG service, when BM25 retrieval is done through sparse vectors from DB.")
                else:
                    raise ValueError("The metadata descriptions cannot be generated because langchain_documents are not loaded in RAG service.")
            metadata_descriptions = RagFilteringMetadataHelper.auto_generate_metadata_descriptions_from_docs_metadata(rag.langchain_documents, 30)
            
        self.metadata_descriptions = metadata_descriptions
        pre_treatment_class = self.workflow_concrete_classes['RAGPreTreatment']
        pre_treatment_class.metadata_descriptions = metadata_descriptions #TODO: might be needed to refactor as a singleton for setting app specific filters values?
        pre_treatment_class.default_filters = self.default_filters        #TODO: might be needed to refactor as a singleton for setting app specific filters values?
        self.tools: list = tools

        # if tools and any(tools):
        #     python_repl = PythonREPL()
        #     repl_tool = Tool(
        #         name="python_repl",
        #         description="Run arbitrary Python code. Get the output with: `print(...)`.",
        #         func=python_repl.run,
        #     )
        #     all_tools = [repl_tool]            
        #     additionnal_tools = [format_tool_to_openai_function(t) for t in tools]
        #     all_tools.extend(additionnal_tools)
        #     self.rag.llm = self.rag.llm.bind_functions(all_tools)
        #     self.tool_executor = ToolExecutor(all_tools)
        
    def set_workflow_concrete_classes(self, override_workflow_available_classes):
        if self.workflow_concrete_classes:
            return
        
        if override_workflow_available_classes:
            # IoC: inject overridden classes for the pipeline
            self.workflow_concrete_classes = override_workflow_available_classes
        else:
            # Set default classes for the pipeline
            self.workflow_concrete_classes = {
                'RAGGuardrails': RAGGuardrails,
                'RAGPreTreatment': RAGPreTreatment,
                'RagRetrieval': RagRetrieval,
                'RAGAugmentedGeneration': RAGAugmentedGeneration,
                'RAGPostTreatment': RAGPostTreatment
            }             
    
    async def run_pipeline_dynamic_streaming_async(self, query: Union[str, Conversation], include_bm25_retrieval: bool = False, give_score=True, pipeline_config_file_path: str = 'rag_pipeline_default_config_wo_AG_for_streaming.yaml', format_retrieved_docs_function = None, all_chunks_output:list = []):
        """Run the full rag inference pipeline: use dynamic pipeline until augmented generation which is streamed async"""
        try:
            analysed_query, retrieved_chunks_list = await self.run_pipeline_dynamic_but_augmented_generation_async(query, include_bm25_retrieval, give_score, pipeline_config_file_path, format_retrieved_docs_function)
        except EndWorkflowException as ex:
            yield ex.message
            return
        except Exception as ex:
            yield str(ex)
            return
        
        retrieved_chunks = retrieved_chunks_list[0]
        async for chunk in self.workflow_concrete_classes['RAGAugmentedGeneration'].rag_augmented_answer_generation_streaming_async(self.rag, query, retrieved_chunks, analysed_query, format_retrieved_docs_function):
            all_chunks_output.append(chunk)
            yield chunk
            
    async def run_pipeline_dynamic_but_augmented_generation_async(self, query: Union[str, Conversation], include_bm25_retrieval: bool = False, give_score=True, pipeline_config_file_path: str = 'rag_pipeline_default_config_wo_AG_for_streaming.yaml', format_retrieved_docs_function = None):
        config = Ressource.load_ressource_file(pipeline_config_file_path, Ressource.rag_configs_package_name)
        workflow_executor = WorkflowExecutor(config, self.workflow_concrete_classes)      
        kwargs_values = {
            'rag': self.rag,
            'query': query,
            'include_bm25_retrieval': include_bm25_retrieval,
            'give_score': give_score,
            'format_retrieved_docs_function': format_retrieved_docs_function }      
        try:
            guardrails_result, retrieved_chunks = await workflow_executor.execute_workflow_async(kwargs_values=kwargs_values)
        
        except EndWorkflowException as ex:
            raise ex
        except Exception as ex:
            raise Exception(f"Error in the workflow: {str(ex)}")

        self.check_for_guardrails(guardrails_result[0])
        analysed_query = kwargs_values['analysed_query']
        return analysed_query, retrieved_chunks

    # Main workflow using the static pipeline
    async def run_pipeline_static_streaming_async(self, query: Union[str, Conversation], include_bm25_retrieval: bool = False, give_score=True, format_retrieved_docs_function = None):
        """Run the full hardcoded rag inference pipeline but async and with streaming LLM augmented generation response"""
        # Run both functions in parallel, where the second one is treated as streaming
        async for idx, result in Execute.run_several_functions_as_concurrent_async_tasks(
            (RAGGuardrails.guardrails_query_analysis, (query)),  # Guardrails check: query analysis
            (self.run_static_inference_pipeline_but_guardrails_async, (), {
                'query': query,
                'include_bm25_retrieval': include_bm25_retrieval,
                'give_score': give_score,
                'format_retrieved_docs_function': format_retrieved_docs_function
            }),
            functions_with_streaming_indexes=[1]  # Indicating that 'run_static_inference_pipeline_but_guardrails_async' is a streaming async function
        ):
            if idx == 1:
                # Yield the streaming chunks of inference pipeline
                async for chunk in result:
                    yield chunk
            elif idx == 0:
                # Guardrails result, check it
                self.check_for_guardrails(result)

    async def run_pipeline_dynamic_no_streaming_async(self, query: Union[str, Conversation], include_bm25_retrieval: bool = False, give_score=True, pipeline_config_file_path: str = 'rag_pipeline_default_config_full_no_streaming.yaml', format_retrieved_docs_function = None) -> str:
        """Run the full rag inference pipeline without streaming async"""
        config = Ressource.load_ressource_file(pipeline_config_file_path, Ressource.rag_configs_package_name)
        workflow_executor = WorkflowExecutor(config, self.workflow_concrete_classes)
        
        kwargs_values = {
            'rag': self.rag,
            'query': query,
            'include_bm25_retrieval': include_bm25_retrieval,
            'give_score': give_score,
            'format_retrieved_docs_function': format_retrieved_docs_function }
        
        try:
            results = await workflow_executor.execute_workflow_async(kwargs_values=kwargs_values)
        except EndWorkflowException as ex:
            raise ex
        except Exception as ex:
            raise Exception(f"Error in the workflow: {str(ex)}")
        return results
        
    async def run_static_inference_pipeline_but_guardrails_async(self, query:Union[str, Conversation], include_bm25_retrieval: bool = False, give_score=True, format_retrieved_docs_function = None):
        """Run the full rag inference pipeline, but without guardrails"""
        # Pre-treatment
        analysed_query, metadata = RAGPreTreatment.rag_static_pre_treatment(self.rag, query)

        # Data Retrieval
        retrieved_chunks = RagRetrieval.hybrid_retrieval_langchain_async(self.rag, query, metadata, include_bm25_retrieval, False, True, give_score)

        # Augmented Answer Generation
        async for chunk in RAGAugmentedGeneration.rag_augmented_answer_generation_streaming_async(self.rag, query, retrieved_chunks, analysed_query, format_retrieved_docs_function):
            yield chunk
    
    def check_for_guardrails(self, guardrails_result:bool):
        """Raise an error if the query has been rejected by guardrails"""
        if not guardrails_result:
            raise Exception("Query rejected by guardrails")
        
    # To test: async wrapper may no works properly
    def run_pipeline_dynamic_streaming_sync(self,
                            query: Union[str, Conversation],
                            include_bm25_retrieval: bool = False,
                            give_score: bool = True,
                            pipeline_config_file_path: str = 'rag_pipeline_default_config_wo_AG_for_streaming.yaml',
                            format_retrieved_docs_function=None,
                            all_chunks_output:list = []) -> Generator:
        sync_generator = Execute.async_generator_wrapper_to_sync(
            self.run_pipeline_dynamic_streaming_async,
            query,
            include_bm25_retrieval,
            give_score,
            pipeline_config_file_path,
            format_retrieved_docs_function,
            all_chunks_output
        )
        for chunk in sync_generator:
            yield chunk
    
    def run_pipeline_static(self,
                            query: Union[str, Conversation],
                            include_bm25_retrieval: bool = False,
                            give_score: bool = True,
                            format_retrieved_docs_function=None) -> tuple:
        sync_generator = Execute.async_generator_wrapper_to_sync(
            self.run_pipeline_static_streaming_async,
            query,
            include_bm25_retrieval,
            give_score,
            format_retrieved_docs_function
        )
        return ''.join(chunk for chunk in sync_generator)
    
