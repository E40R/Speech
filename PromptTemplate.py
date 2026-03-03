from langchain_core.prompts import ChatPromptTemplate #v0.2+ has prompts in core 
from langchain_core.output_parsers import PydanticOutputParser
from Services.triage_service import TriageService

parser = PydanticOutputParser(pydantic_object=TriageService)

# System Message
system_message = """ You=triage assistant, Do NOT diagnose.Always return valid JSON only. If human input is non health concern conversation then just rewrite exact input in key_details without changing."""


# Create prompt template
prompt = ChatPromptTemplate(
    [("system", system_message), ("human", """input:{user_input}. parser:{format_instructions}""")],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

tts_prompt = ChatPromptTemplate(
    [("system","""You are a calm veterinary assistant speaking to a pet owner. Generate a short conversational response suitable for text-to-speech TTS. Rules: - Do NOT diagnose. - Do NOT change risk level. - Do NOT add new medical reasoning.- Only convert structured triage data into natural, reassuring speech. - If risk_level is emergency, emphasize urgency clearly but calmly."""),
     ("human", """{triage_input}.""")]
)
