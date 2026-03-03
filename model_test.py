from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from PromptTemplate import prompt, parser, tts_prompt
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

#1 stt/vad
user_input="Hello mybot how are you!"


#2 Triagellm:done
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
chain1 = prompt | llm | parser
result1 = chain1.invoke({
    "user_input": user_input
})


#3 JSON File: done
data = result1.model_dump() #convert Pydantic model to dict
#create outputs folder if it doesn't exist : or else append code: os.makedirs("outputs", exist_ok=True) filename = "outputs/triage_results.json" # Convert structured result to dict entry = result.model_dump() # Add metadata entry["user_input"] = user_input entry["timestamp"] = datetime.now().isoformat() # Load existing file if present if os.path.exists(filename): with open(filename, "r", encoding="utf-8") as f: try: data = json.load(f) except json.JSONDecodeError: data = [] else: data = [] # Append new entry data.append(entry) # Write back to file with open(filename, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"outputs/triage_result_{timestamp}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)



#4 Output tts llm: done
#can send user input directly or notedown key points in user input in triage and send only triage: now choosen both to handle even conversations: handled conv by telling llm1 to keep it in key details
llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",   # cheap + good quality
    temperature=0.3)
def generate_tts_response(triage_result):
    chain = tts_prompt | llm2
    response = chain.invoke(str(triage_result))
    return response.content
print(generate_tts_response(result1))

#5 TTS



#6 Log and append to conversation history : need to change code to add AI Message human message ect for history.