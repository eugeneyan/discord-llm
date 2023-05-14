"""
Agent that can query a database
Docs: https://langchain.readthedocs.io/en/latest/modules/chains/examples/sqlite.html
Dataset: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
"""
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.agents import (AgentExecutor, Tool, ZeroShotAgent,
                              create_sql_agent)
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase

from config import SQL_MODEL
from logger import logger
from utils import prettify_agent_response, prettify_chain_response, timer

# Load env
load_dotenv()
DB = SQLDatabase.from_uri('sqlite:///data/books.db')
TOOLKIT = SQLDatabaseToolkit(db=DB)
TOOLS = TOOLKIT.get_tools()
TOOL_NAMES = [tool.name for tool in TOOLS]


PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}"""


# Defines agent to query a database
@timer
def sql_agent(query: str, temperature: float = 0, top_k: int = 10, model: str = SQL_MODEL) -> str:
    """
    Create Agent that can query a database.
    """
    # Write prompt and create zero-shot agent
    prompt = ZeroShotAgent.create_prompt(
        tools=TOOLKIT.get_tools(),
        prefix=PREFIX.format(dialect=TOOLKIT.dialect, top_k=top_k),
        suffix=SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        input_variables=['input', 'agent_scratchpad']
    )
    logger.info(prompt.template)

    # Create LLM and call API
    llm = ChatOpenAI(temperature=temperature, model_name=model)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Create agent with tools
    agent = ZeroShotAgent(llm_chain=llm_chain,
                          tools=TOOLS, tool_names=TOOL_NAMES)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=TOOLS, max_iterations=10,
                                                        verbose=True, return_intermediate_steps=True)

    response = agent_executor({'input': query})
    pretty_response = prettify_agent_response(response)

    return pretty_response


# Create chain to query database
@timer
def sql_chain(query: str, temperature: float = 0, model: str = SQL_MODEL) -> str:
    """
    Create chain that can query a database.
    """
    llm = OpenAI(temperature=temperature, model_name=model)
    db_chain = SQLDatabaseChain(
        llm=llm, database=DB, verbose=True, return_intermediate_steps=True)

    response = db_chain(query)
    logger.info(f'Response: {response}')
    pretty_response = prettify_chain_response(response)
    return pretty_response
