import sys
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama  import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor 
from tools import vet_instagram_creator_for_promotion

load_dotenv()

# The LLM's output contains unicode chars (e.g. \u2011) that Windows' default
# cp1252 console encoding can't print, which crashes every print() call.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _ensure_ollama_running() -> None:
    """Fail fast with a clear message if the Ollama server isn't reachable."""
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3).raise_for_status()
    except requests.RequestException:
        raise SystemExit(
            "Could not reach Ollama at http://localhost:11434.\n"
            "Start it with `ollama serve` (or the Ollama desktop app) and re-run."
        )


class Reacherreponse(BaseModel):
    topic : str
    summary: str
    sources : list[str]
    tools_used : list[str]


parser = PydanticOutputParser(pydantic_object=Reacherreponse)

llm  = ChatOllama(model="gpt-oss:20b")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [vet_instagram_creator_for_promotion]

agent = create_tool_calling_agent(
    llm = llm,
    prompt=  prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
_ensure_ollama_running()

raw = input(
    "Enter Instagram usernames to vet, comma-separated (e.g. maniesh6900, anotheruser): "
).strip()
usernames = [u.strip() for u in raw.split(",") if u.strip()] or ["maniesh6900"]

for username in usernames:
    print(f"\n=== Vetting @{username} ===")
    query = f"Vet the Instagram creator @{username} for the product: Logitech G102 Light Sync Gaming Mouse"
    res = agent_executor.invoke({"query": query})
    try:
        paresed_res = parser.parse(res.get("output"))
        print(paresed_res)
    except Exception as e:
        print("ERROR WHILE PERSERING THE RAW REPONSE, RAW REPONSE : ", res)