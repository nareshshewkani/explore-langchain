import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()


def main():
    print("Hello from langchain-course!")

    info = """
    John Patrick Ternus (born May 1975) is an American engineer and business executive who has been the chief executive officer (CEO) of Apple since September 1, 2026. He has also served on the company's board of directors since that date. Ternus joined Apple's product design team in 2001 and was the company's senior vice president of hardware engineering from 2021 to 2026.

    Apple has credited Ternus with work on the iPad and AirPods product lines, and he oversaw hardware engineering during the company's transition from Intel processors to Apple silicon in Mac computers.
    """

    summary_template = """
    given the information {information} about a person I want to create
    1) A short summary about that person
    2) 2 Interesting facts about that person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model='gpt-4')

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": info})
    print(response.content)




if __name__ == "__main__":
    main()
