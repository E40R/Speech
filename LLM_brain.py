from langchain_perplexity import ChatPerplexity
from langchain_core.messages import HumanMessage, AIMessage
from PromptTemplate import prompt, parser, tts_prompt
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

# ── Conversation state ────────────────────────────────────────────────────────

conversation_history = []
previous_key_details = []


# ── LLM functions ─────────────────────────────────────────────────────────────

def triagellm(user_input: str):
    llm = ChatPerplexity(model="sonar", temperature=0)
    chain = prompt | llm | parser
    result = chain.invoke({
        "user_input": user_input,
        "previous_context": previous_key_details
    })
    return result


def generate_tts_response(triage_result) -> str:
    llm2 = ChatPerplexity(model="sonar", temperature=0.3)
    chain = tts_prompt | llm2
    response = chain.invoke(str(triage_result))
    return response.content


# ── History update ────────────────────────────────────────────────────────────

def update_history(user_input: str, tts_text: str, triage_result):
    global previous_key_details
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=tts_text))
    previous_key_details = list(set(previous_key_details + triage_result.key_details))  #addedd as set/list to include past key details 


# ── JSON logger ───────────────────────────────────────────────────────────────

def log_to_json(user_input: str, triage_result):
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/triage_result_{timestamp}.json"
    data = triage_result.model_dump()
    data["user_input"] = user_input
    data["timestamp"] = datetime.now().isoformat()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
