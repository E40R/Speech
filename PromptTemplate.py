from langchain_core.prompts import ChatPromptTemplate #v0.2+ has prompts in core 
from langchain_core.output_parsers import PydanticOutputParser
from Services.triage_service import TriageService

parser = PydanticOutputParser(pydantic_object=TriageService)

# System Message
system_message = """ 
You=triage assistant, Do NOT diagnose.Always return valid JSON only. 
If previous context exists, use it to maintain continuity: {previous_context}
If human input is non health concern conversation then just store exact input in key_details without changing.
"""


# Create prompt template
prompt = ChatPromptTemplate(
    [
        ("system", system_message),
        ("human", """input:{user_input}. parser:{format_instructions}"""),
    ],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

tts_prompt = ChatPromptTemplate(
    [
        ("system","""You are a calm veterinary assistant speaking to a pet owner. Generate a conversational response suitable for TTS. 
        Rules:- Do NOT diagnose/definitive diagnoses. 
        -Include disclaimers about not replacing professional vet care. - Do NOT add or change anything new.
        - Only convert structured triage data into natural, reassuring speech.
        - If triage has conversation input then give polite and short response like "hey how are you?!" and try to keep response small and avoid disclaimers in only if 'converational input' else give the disclaimer.  
         - If risk_level is emergency, emphasize urgency clearly."""),
        ("human", """{triage_input}.""")
    ]
)
