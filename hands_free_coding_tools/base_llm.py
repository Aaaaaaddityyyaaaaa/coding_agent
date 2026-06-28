from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

API_GROQ = os.getenv("GROQ_API_KEY")

def get_llm(llm_type: str = "brain"):
    if llm_type == "brain":
        return ChatGroq(api_key=API_GROQ, model="openai/gpt-oss-120b", temperature=0.1)
    else:
        return ChatGroq(api_key=API_GROQ, model="moonshotai/kimi-k2-instruct-0905", temperature=0.2)
