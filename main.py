from dotenv import load_dotenv



from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
# from tavily import TavilyClient
from langchain_tavily import TavilyResearch, TavilySearch, TavilyCrawl, TavilyExtract, TavilyGetResearch, TavilyMap

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

def main():

    print("Hello. Welcome to Search Agent!")
    llm = ChatOllama(model="gemma4:12b", reasoning=True)
    # tools = [search]
    toolbox = [TavilyResearch(), TavilySearch(),TavilyCrawl(),  TavilyExtract(), TavilyGetResearch(), TavilyMap()]
    agent = create_agent(model=llm, tools=toolbox)
    # result = agent.invoke({"messages":HumanMessage("can you research for 3 job openings in Mumbai and Pune region. I know Agentic AI and Full Stack development using SpringBoot and Angular. Give me job links also to apply. Choose the best 3 job openings.")})
    result = agent.invoke({"messages":HumanMessage("you have access to bunch of tools. check if gpt-6 is released now and if yes clcan u research about gpt-6. and create a detailed report about it's comparision with gpt-5, gpt-4, and 10 other models which are peak of every AI company like claude. You may use tools back and forth but come up with very researched answers. You may take your time thats okay.")})
    print(result)


if __name__ == "__main__":
    main()
