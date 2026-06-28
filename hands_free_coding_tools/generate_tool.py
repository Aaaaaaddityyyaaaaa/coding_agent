from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from hands_free_coding_tools.base_llm import get_llm
import os
import re
from dotenv import load_dotenv

load_dotenv()

code_gen_llm = get_llm(llm_type="code")


def extract_code(response: str, language: str = "python") -> str:
    pattern = rf"```(?:{language})?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return response.strip()


@tool
def generate_tool(prompt: str) -> str:
    """Generates code based on the given prompt. Returns only the code block."""
    messages = [
        SystemMessage(content=(
            "You are a code generation tool. "
            "Generate only the requested code with no explanation or preamble. "
            "Wrap your output in a ```python ... ``` block."
        )),
        HumanMessage(content=prompt),
    ]
    response = code_gen_llm.invoke(messages).content
    return extract_code(response, "python")
