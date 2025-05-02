import asyncio
from typing import Optional, Union
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
#
from common_tools.helpers.llm_helper import Llm
from common_tools.helpers.ressource_helper import Ressource
from common_tools.models.conversation import Conversation
from common_tools.models.question_analysis_base import QuestionAnalysisBase
from rag_common_tools.rag_service import RagService
from common_tools.helpers.execute_helper import Execute
from common_tools.helpers.method_decorator_helper import MethodDecorator

class RAGAugmentedGeneration:
    augmented_generation_prompt:str = None
    
    @staticmethod
    def rag_augmented_answer_generation_no_streaming_sync(
        rag: RagService, 
        query: Union[str, Conversation], 
        retrieved_chunks: list, 
        analysed_query: QuestionAnalysisBase, 
        format_retrieved_docs_function=None
    ):
        all_chunks = []
        sync_generator = Execute.async_generator_wrapper_to_sync(RAGAugmentedGeneration.rag_augmented_answer_generation_streaming_async, rag, query, retrieved_chunks, analysed_query, True, all_chunks, format_retrieved_docs_function)
        for chunk in sync_generator:
            pass
        return ''.join(chunk for chunk in Llm.get_text_from_chunks(all_chunks))

    @staticmethod
    async def rag_augmented_answer_generation_no_streaming_async(
            rag: RagService,
            query: Union[str, Conversation],
            retrieved_chunks: list,
            analysed_query: QuestionAnalysisBase,
            format_retrieved_docs_function: any = None
        ) -> any:
            chunks: list[str] = []
            async for output in RAGAugmentedGeneration.rag_augmented_answer_generation_streaming_async(
                rag,
                query,
                retrieved_chunks,
                analysed_query,
                True,
                chunks,
                format_retrieved_docs_function
            ):
                pass
            return ''.join(chunks)

    @staticmethod
    @MethodDecorator.print_func_execution_infos()
    async def rag_augmented_answer_generation_streaming_async(rag: RagService, query:Union[str, Conversation], retrieved_chunks: list, analysed_query: QuestionAnalysisBase, is_stream_decoded = False, all_chunks_output: list[str] = [], function_for_specific_formating_retrieved_docs = None):
        # Select the smallest llm for the augmented generation task, as it takes lots of tokens
        async for chunk in RAGAugmentedGeneration.augmented_answer_generation_from_llm_streaming_async(rag.llm_1, query, retrieved_chunks, analysed_query, is_stream_decoded, all_chunks_output, function_for_specific_formating_retrieved_docs):
            yield chunk
    
    @staticmethod
    async def augmented_answer_generation_from_llm_streaming_async(llm_or_chain: Runnable, query:Union[str, Conversation], retrieved_chunks: list, analysed_query: QuestionAnalysisBase, is_stream_decoded = False, all_chunks_output: list[str] = [], function_for_specific_formating_retrieved_docs = None):
        if retrieved_chunks and any(retrieved_chunks) and isinstance(retrieved_chunks[0], tuple): 
            retrieved_chunks = [doc[0] for doc in retrieved_chunks] # Remove scores if present

        if not RAGAugmentedGeneration.augmented_generation_prompt:
            RAGAugmentedGeneration.augmented_generation_prompt = Ressource.get_rag_augmented_generation_prompt_generic()
        augmented_generation_prompt = RAGAugmentedGeneration.augmented_generation_prompt

        question_w_history = Conversation.conversation_history_as_str(query)
        augmented_generation_prompt = augmented_generation_prompt.replace("{question}", question_w_history)
        
        # handle request for respond in a specific language
        additional_instructions = ''
        if hasattr(analysed_query, 'detected_language') and not analysed_query.detected_language.__contains__("english"):
            additional_instructions = Ressource.get_prefiltering_translation_instructions_prompt()
            additional_instructions = additional_instructions.replace("{target_language}", analysed_query.detected_language)
        augmented_generation_prompt = augmented_generation_prompt.replace("{additional_instructions}", additional_instructions)
        
        rag_custom_prompt = ChatPromptTemplate.from_template(augmented_generation_prompt)

        if function_for_specific_formating_retrieved_docs is None:
            context = retrieved_chunks
        else:
            context = function_for_specific_formating_retrieved_docs(retrieved_chunks)
        
        rag_chain = rag_custom_prompt | llm_or_chain | RunnablePassthrough()

        async for chunk in Llm.invoke_as_async_stream('RAG augmented generation', rag_chain, context):
            chunk_final = chunk if (not is_stream_decoded or isinstance(chunk, str)) else chunk.decode('utf-8').replace(Llm.new_line_for_stream_over_http, '\n')
            all_chunks_output.append(chunk_final)
            yield chunk_final 
