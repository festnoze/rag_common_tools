import asyncio
import time
from typing import Optional, Union
from common_tools.helpers.txt_helper import txt
from common_tools.models.conversation import Conversation
from common_tools.workflows.workflow_output_decorator import workflow_output

class RAGGuardrails:
    # @staticmethod
    # @workflow_output('guardrails_result')
    # def guardrails_query_analysis(query:Union[str, Conversation]) -> bool:
    #     time.sleep(0.5) #todo: to implement
    #     user_query = Conversation.get_user_query(query)
    #     if user_query == "bad query":
    #         return False
    #     return True
    
    @staticmethod
    @workflow_output('guardrails_result')
    async def guardrails_query_analysis_async(query:Union[str, Conversation]) -> bool:
        #TODO: FAKED: Add real implementation!!!
        await asyncio.sleep(1.5) 
        user_query = Conversation.get_user_query(query)
        if user_query == "bad query":
            return False
        return True
