from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langsmith import traceable
load_dotenv()
MAX_ITERATIONS = 10
MODEL = 'qwen3.5:4b-mlx'

@tool
def get_product_price(product_name:str)->int:
    """
    lookup for the actual price of product

    Args:
    product_name: the actual name of product (case-sensitive)
    
    Returns:
    The price of the product
    """

    product_price_list = {
        "laptop": 1000,
        "mouse": 60,
        "keyboard":100,
        "desktop": 1200
    }

    return product_price_list.get(product_name, 50)

@tool
def apply_discount(price:int, tier:str)->int:
    """
    applies discount on a product price based on the customer tier.

    Args:
    1) price: The original price of the product

    2) tier: The customer tier that decides the discount the user will get.
    Must be one of the following three values [BRONZE, GOLD, SILVER] (case-insensitive)

    Returns:
    The final price of the product after applying discount based on tier
    """
    discount_tier_map = {
        "BRONZE": 5,
        "GOLD": 12,
        "SILVER":7
    }

    discount_pct = discount_tier_map.get(tier, 0)
    final_price = price * ((100-discount_pct)/100)
    return final_price

@traceable(name="agents-under-the-hood")
def run_agent(question:str):
    print("Hello There!. We are running agent loop")
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    llm_with_tools = llm.bind_tools(tools)

    print(f" question: {question}")
    print("="*60)
    print()

    message = [
        SystemMessage(
            content = (
                "You are a Shopping Assistant agent." 
                "You have access to a product catalog tool and a discount tool"
            )
        ),
        HumanMessage(
            content= question
        )
    ]

    final_answer = "MAX ITERATIONS EXCEEDED. COULDN'T REACH A SOLUTION"
    for i in range(1, MAX_ITERATIONS+1):
        print(f"===ITERATION=== {i}")
        ai_message = llm_with_tools.invoke(message)
        if ai_message.tool_calls:
            tool_calls = ai_message.tool_calls
            # For simplicity, we will take 1 tool call (first one). Multiple tool calls means parallel tool calling, many LLMs do that.
            tool_call = tool_calls[0]
            # calling the tool with the LLM provided arguements.
            tool_name = tool_call.get("name")
            tool_call_id = tool_call.get("id")
            tool_args = tool_call.get("args")

            # execute the tool
            tool_to_use = tools_dict.get(tool_name)
            observation = tool_to_use.invoke(tool_args)
            print(observation)
            message.append(ai_message)
            message.append(ToolMessage(content=observation, tool_call_id = tool_call_id))


        else:
            # If no tool call, we can return the final answer
            final_answer = ai_message.content
            break


    return final_answer
    
    





if __name__== "__main__":
    print("Welcome to the agent!")
    question = "What will be the price of a laptop with gold tier membership?"
    print(run_agent(question))








