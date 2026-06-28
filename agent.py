from langchain_groq import ChatGroq 
from langchain_core.tools import tool
from pydantic import BaseModel , Field
from langgraph.graph import StateGraph

from dotenv import load_dotenv
import os

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY") 

class CabDetails(BaseModel)  :
  src : str = Field(description="The starting point from where the driver picksup the customer")
  dest  : str =Field(description="The end point  where the driver has to drop the customer")
  passengers : int = Field(default=1 , description="The no. of passengers that will be travelling")

@tool(args_schema=CabDetails)
def book_a_cab (src:str , dest : str , passengers:int)->str : 
  """The tool books acab for the user from the sourcce to destinantion it searches the database finds the driver and helps book a cab"""
  return{"src" : src , "dest"  :dest , "passengers":passengers}

llm = ChatGroq(api_key=  GROQ_KEY , model="llama-3.3-70b-versatile" , temperature=1)
llm_with_tools = llm.bind_tools([book_a_cab])
response=llm_with_tools.invoke("can you book me a hotel")
print(response.tool_calls)