from langchain_google_genai import ChatGoogleGenerativeAI

from PromptTemplate import prompt, parser, tts_prompt
from STT_service import listen
from TTS_service import speak
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

#1 stt/vad
user_input=listen()


#2 Triagellm:done

def triagellm(user_input):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chain1 = prompt | llm | parser
    result1 = chain1.invoke({"user_input": user_input})
    return result1
result1= triagellm(user_input)


#3 JSON File: done



#4 Output tts llm: done
#can send user input directly or notedown key points in user input in triage and send only triage: now choosen both to handle even conversations: handled conv by telling llm1 to keep it in key details

def generate_tts_response(triage_result):
    llm2 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",   # cheap + good quality
        temperature=0.3)
    chain = tts_prompt | llm2
    response = chain.invoke(str(triage_result))
    return response.content

result2=generate_tts_response(result1)
print(result2)

#5 TTS
#two options Google Cloud TTS or Deepgram Aura

response = generate_tts_response(result2)
speak(response)  # replaces print(response)



#6 Log and append to conversation history : need to change code to add AI Message human message ect for history.
