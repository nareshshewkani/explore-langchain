from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents.factory import ToolStrategy # Or standard import location for your framework version

# from tavily import TavilyClient
from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyGetResearch,
    TavilyMap,
    TavilyResearch,
    TavilySearch,
)
from pydantic import BaseModel, Field

load_dotenv()
# tavily = TavilyClient()

# @tool
# def search(query:str)->str:
#     """
#     This Tool is used for searching the internet
#     Args:
#     query: The query to search for

#     Returns:
#     The Search Result
#     """
#     print(f"Searching the {query}")

#     return tavily.search(query)


class Source(BaseModel):
    """Schema for source url used by agent"""

    url: str = Field("The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field("Agents response to the query")
    sources: List[Source] = Field(default_factory=list, description="List of source urls for jobs")


def main():

    print("Hello. Welcome to Search Agent!")
    llm = ChatOllama(model="gemma4:12b-nvfp4", reasoning=True)
    # structured_llm = llm.with_structured_output(AgentResponse, method="json_mode")

    # 1. Define the graceful tool limiter
    graceful_limiter = ToolCallLimitMiddleware(
        run_limit=20,  # Stop the agent after 20 tool calls in a single prompt
        exit_behavior="end",  # Does not Allow the LLM to write a final fallback message 
    )

    # 2. Inject it into the agent creation

    # tools = [search]
    toolbox = [
        TavilyResearch(),
        TavilySearch(),
        TavilyCrawl(),
        TavilyExtract(),
        TavilyGetResearch(),
        TavilyMap(),
    ]
    agent = create_agent(model=llm, tools=toolbox, middleware=[graceful_limiter], response_format=ToolStrategy(AgentResponse))
    result = agent.invoke(
        {
            "messages": HumanMessage(
                "can you search for 3 job openings in Mumbai and Pune region along with JD and link to apply. I know Agentic AI and Full Stack development using SpringBoot and Angular. Choose the best 3 job openings and share the JD and link"
            )
        }
    )
    # result = agent.invoke({"messages":HumanMessage("you have access to bunch of tools. check if gpt-6 is released now and if yes clcan u research about gpt-6. and create a detailed report about it's comparision with gpt-5, gpt-4, and 10 other models which are peak of every AI company like claude. You may use tools back and forth but come up with very researched answers. You may take your time thats okay.")})
    print(result)


if __name__ == "__main__":
    main()
