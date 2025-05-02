from common_tools.helpers.ressource_helper import Ressource
from common_tools.workflows.end_workflow_exception import EndWorkflowException

class GreetingsEndsPipelineException(EndWorkflowException):
    def __init__(self):
        message = Ressource.load_ressource_file('greetings_default_message.txt')
        super().__init__("_salutations_", message)