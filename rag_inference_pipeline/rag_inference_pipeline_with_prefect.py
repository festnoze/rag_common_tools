# from langchain.tools import tool
# from langchain_core.documents import Document
# import time
# from langchain.indexes import VectorstoreIndexCreator
# from langgraph.prebuilt.tool_executor import ToolExecutor
# from langchain.tools.render import format_tool_to_openai_function
# from langchain.agents import Tool
# from langchain_experimental.utilities import PythonREPL
# from prefect import flow, task
# from prefect.task_runners import ConcurrentTaskRunner, ThreadPoolTaskRunner
# from prefect.logging import get_run_logger

# from common_tools.helpers.llm_helper import Llm
# from common_tools.helpers.ressource_helper import Ressource
# from common_tools.models.logical_operator import LogicalOperator
# from common_tools.models.question_analysis import QuestionAnalysis, QuestionAnalysisPydantic
# from common_tools.helpers.rag_filtering_metadata_helper import RagFilteringMetadataHelper
# from rag_common_tools.rag_service import RagService
# from rag_common_tools.rag_inference_pipeline.rag_inference_pipeline import RagInferencePipeline
# from langchain_core.documents import Document
# from langchain_core.runnables import Runnable, RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate

# class RagInferencePipelineWithPrefect:
    
#     def __init__(self, rag: RagService, default_filters: dict = {}, tools: list = None):
#         self.rag: RagService = rag
#         self.default_filters = default_filters
#         self.tools = tools

#         if tools and any(tools):
#             python_repl = PythonREPL()
#             repl_tool = Tool(
#                 name="python_repl",
#                 description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#                 func=python_repl.run,
#             )

#             # Pass the tool to the agent supervisor & to the tool executor
#             all_tools = [repl_tool]
            
#             additionnal_tools = [format_tool_to_openai_function(t) for t in tools] #TODO: check compatibility out of OpenAI
#             all_tools.extend(additionnal_tools)
#             self.rag.llm = self.rag.llm.bind_functions(all_tools)
#             self.tool_executor = ToolExecutor(all_tools)


#     # Main flow
#     @flow(
#             name="rag inference pipeline flow", 
#             task_runner=ThreadPoolTaskRunner(max_workers=10),
#             retries=0,
#             log_prints=True)
#     def run(self, query: str, include_bm25_retrieval: bool = False, give_score = True):
#         logger = get_run_logger()
#         logger.info("Starting rag inference pipeline")

#         guardrails_result = self.guardrails_query_analysis.submit(query)
        
#         # Pre-treatment query: translate, search for meta-data
#         analysed_query, metadata = self.rag_pre_treatment(query)
        
#         # Data Retrieval subflow (runs in parallel for rag and BM25)
#         retrieved_chunks = self.rag_hybrid_retrieval(analysed_query, metadata, include_bm25_retrieval, give_score)

#         # Augmented Answer Generation subflow
#         response = self.rag_augmented_answer_generation_no_streaming_sync(retrieved_chunks, analysed_query)

#         # Post-treatment subflow
#         final_response = self.rag_post_treatment(response)
        
#         # Wait for guardrails analysis to finish        
#         guardrails_result = guardrails_result.result()
#         if not guardrails_result:
#             print("Query rejected by guardrails")
#             return "I cannot answer your question, because its topic is explicitly forbidden.", retrieved_chunks #todo: translate when needed

#         min_score = 0.5
#         if give_score and min(chunk[1] for chunk in retrieved_chunks) > min_score:
#             print("Query rejected by guardrails")
#             return "I cannot answer your question, as I found no relevant information about your query.", retrieved_chunks
        
#         return final_response, retrieved_chunks
    
#     # Pre-treatment subflow
#     @flow(name="rag pre-treatment", task_runner=ThreadPoolTaskRunner(max_workers=3))
#     def rag_pre_treatment(self, query):
#         questionAnalysis = self.analyse_query_language(query)
#         extracted_metadata = self.extract_explicit_metadata(questionAnalysis.translated_question)
        
#         questionAnalysis.translated_question = extracted_metadata[0].strip()
#         metadata = extracted_metadata[1]

#         return questionAnalysis, metadata
    
#     @task(log_prints=True)
#     def guardrails_query_analysis(self, query) -> bool:
#         # Logic for guardrails analysis (returns True if OK, False if rejected)
#         time.sleep(5)
#         if "bad query" in query:  # Example check
#             return False
#         #print(">>> Query accepted by guardrails")
#         return True
    
#     @task
#     def analyse_query_language(self, query):
#         prefilter_prompt = Ressource.get_language_detection_prompt()
#         prefilter_prompt = prefilter_prompt.replace("{question}", query)
#         prompt_for_output_parser, output_parser = Llm.get_prompt_and_json_output_parser(prefilter_prompt, QuestionAnalysisPydantic, QuestionAnalysis)
#         response = Llm.invoke_parallel_prompts_with_parser_batchs_fallbacks("rag prefiltering", [self.rag.llm, self.rag.llm], output_parser, 10, *[prompt_for_output_parser])
#         questionAnalysis = response[0]
#         questionAnalysis['question'] = query
#         if questionAnalysis['detected_language'].__contains__("english"):
#             questionAnalysis['translated_question'] = query
#         return QuestionAnalysis(**questionAnalysis)
    
#     @task
#     def extract_explicit_metadata(self, query) -> tuple[str, dict]:
#         filters = {}
#         if RagFilteringMetadataHelper.has_manual_filters(query):
#             filters, query = RagFilteringMetadataHelper.extract_manual_filters(query)
#         else:
#             filters = self.default_filters       
#         return query, filters

#     # Data Retrieval sub-flow with parallel rag and BM25 retrieval
#     @flow(name="rag hybrid retrieval", task_runner=ThreadPoolTaskRunner(max_workers=3))
#     def rag_hybrid_retrieval(self, analysed_query: QuestionAnalysis, metadata, include_bm25_retrieval: bool = False, give_score: bool = True, max_retrived_count: int = 10):
#         rag_retrieved_chunks = self.semantic_vector_retrieval.submit(analysed_query, metadata, give_score, max_retrived_count)
#         if include_bm25_retrieval:
#             bm25_retrieved_chunks = self.bm25_retrieval.submit(analysed_query.translated_question, metadata, give_score, max_retrived_count)

#         rag_retrieved_chunks = rag_retrieved_chunks.result()
#         bm25_retrieved_chunks = bm25_retrieved_chunks.result() if include_bm25_retrieval else []
#         retained_chunks = RagInferencePipelineWithPrefect.hybrid_chunks_selection(rag_retrieved_chunks, bm25_retrieved_chunks, give_score, max_retrived_count)
#         return retained_chunks

#     @task    
#     def hybrid_chunks_selection(rag_retrieved_chunks: list[Document], bm25_retrieved_chunks: list[Document] = None, give_score: bool = False, max_retrived_count: int = None):
#         if not bm25_retrieved_chunks or not any(bm25_retrieved_chunks):
#             return rag_retrieved_chunks
        
#         rag_retrieved_chunks.extend([(chunk, 0) for chunk in bm25_retrieved_chunks] if give_score else bm25_retrieved_chunks)
        
#         if max_retrived_count:
#             if give_score:
#                 rag_retrieved_chunks = sorted(rag_retrieved_chunks, key=lambda x: x[1], reverse=True)
#             rag_retrieved_chunks = rag_retrieved_chunks[:max_retrived_count]

#         return rag_retrieved_chunks
    
#     @task
#     def semantic_vector_retrieval(self, analysed_query: QuestionAnalysis, filters, give_score: bool = False, max_retrived_count: int = 10, min_score: float = None, min_retrived_count: int = None):
#         retrieved_chunks = self.rag.semantic_vector_retrieval(analysed_query.translated_question, None, filters, give_score, max_retrived_count, min_score, min_retrived_count)
#         return retrieved_chunks
    
#     @task
#     def bm25_retrieval(self, analysed_query: QuestionAnalysis, filters: dict, give_score: bool, k = 3):
#         if filters and any(filters):
#             filtered_docs = [doc for doc in self.rag.langchain_documents if RagInferencePipeline.filters_predicate(doc, filters)]
#         else:
#             filtered_docs = self.rag.langchain_documents

#         bm25_retriever = self.rag._build_bm25_retriever(filtered_docs, k)#, filters
#         bm25_retrieved_chunks = bm25_retriever.invoke(analysed_query.translated_question)
       
#         if give_score:
#             score = 0.1
#             return [(doc, score) for doc in bm25_retrieved_chunks]
#         else:
#             return bm25_retrieved_chunks
        
#      # Helper function to evaluate filter(s) (handling nested operators)
#     @staticmethod
#     def filters_predicate(doc, filters, operator=LogicalOperator.AND):
#         # If filters is a dictionary (single filter or operator like $and/$or)
#         if isinstance(filters, dict):
#             if "$and" in filters:
#                 # Recursively handle $and with multiple conditions
#                 return RagInferencePipeline.filters_predicate(doc, filters["$and"], LogicalOperator.AND)
#             elif "$or" in filters:
#                 # Recursively handle $or with multiple conditions
#                 return RagInferencePipeline.filters_predicate(doc, filters["$or"], LogicalOperator.OR)
#             else:
#                 # Handle single key-value filter (field-value pair)
#                 return all(doc.metadata.get(key) == value for key, value in filters.items())

#         # If filters is a list, apply the operator (AND/OR)
#         elif isinstance(filters, list):
#             if operator == LogicalOperator.AND:
#                 return all(
#                     RagInferencePipeline.filters_predicate(doc, sub_filter, LogicalOperator.AND)
#                     for sub_filter in filters
#                 )
#             elif operator == LogicalOperator.OR:
#                 return any(
#                     RagInferencePipeline.filters_predicate(doc, sub_filter, LogicalOperator.OR)
#                     for sub_filter in filters
#                 )
#             else:
#                 raise ValueError(f"Unhandled operator: {operator}")

#         return False
    
    
#     # Augmented answer generation subflow
#     @flow(name="rag answer generation")
#     def rag_augmented_answer_generation_no_streaming_sync(self, retrieved_chunks: list, questionAnalysis: QuestionAnalysis):
#         return self.rag_response_generation(retrieved_chunks, questionAnalysis)

#     @task
#     def rag_response_generation(self, retrieved_chunks: list, questionAnalysis: QuestionAnalysis):
#         # Remove score from retrieved docs
#         retrieved_chunks = [doc[0] if isinstance(doc, tuple) else doc for doc in retrieved_chunks]
#         return self.generate_augmented_response_from_retrieved_chunks(self.rag.llm, retrieved_chunks, questionAnalysis)
        
#     # Post-treatment subflow
#     @flow(name="rag post-treatment")
#     def rag_post_treatment(self, response):
#         return self.response_post_treatment(response)
        
#     @task
#     def response_post_treatment(self, guardrails_answer: bool, rag_answer: str, analysed_query: QuestionAnalysis):
#         if guardrails_answer == True:
#             return rag_answer
#         else:
#             if analysed_query.detected_language == "french":
#                 return "Je ne peux pas répondre à votre question, car son sujet est explicitement interdit."
#             else:
#                 return "I cannot answer your question, because its topic is explicitly forbidden."
    
#     @staticmethod
#     def generate_augmented_response_from_retrieved_chunks(self, llm: Runnable, retrieved_docs: list[Document], questionAnalysis: QuestionAnalysis, format_retrieved_docs_function = None) -> str:
#         retrieval_prompt = Ressource.get_rag_augmented_generation_query_prompt()
#         retrieval_prompt = retrieval_prompt.replace("{question}", questionAnalysis.translated_question)
#         additional_instructions = ''
#         if not questionAnalysis.detected_language.__contains__("english"):
#             additional_instructions = Ressource.get_prefiltering_translation_instructions_prompt()
#             additional_instructions = additional_instructions.replace("{target_language}", questionAnalysis.detected_language)
#         retrieval_prompt = retrieval_prompt.replace("{additional_instructions}", additional_instructions)
#         rag_custom_prompt = ChatPromptTemplate.from_template(retrieval_prompt)

#         if format_retrieved_docs_function is None:
#             context = retrieved_docs
#         else:
#             context = format_retrieved_docs_function(retrieved_docs)
        
#         rag_chain = rag_custom_prompt | llm | RunnablePassthrough()
#         answer = rag_chain.invoke(input= context)

#         return Llm.get_content(answer)  