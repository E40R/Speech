from langchain_core.prompts import ChatPromptTemplate #v0.2+ has prompts in core 
from langchain_core.output_parsers import PydanticOutputParser
from Services.triage_service import TriageService

parser = PydanticOutputParser(pydantic_object=TriageService)

# System Message
system_message = """ 
You=triage assistant, Do NOT diagnose.Always return valid JSON only.
"""


# Create prompt template
prompt = ChatPromptTemplate(
    [
        ("system", system_message),
        ("human", """{user_input} {format_instructions}"""),
    ],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

prompt2= ChatPromptTemplate(
    [
        ("system", system_message),
        ("human", """{user_input}{triage} {format_instructions}"""),
    ],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)