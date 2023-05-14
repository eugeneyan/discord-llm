"""
Module for searching the internet.
"""
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper

from config import SEARCH_MODEL
from logger import logger
from utils import prettify_agent_response, timer

# Create tools
load_dotenv()
search = GoogleSearchAPIWrapper()
TOOLS = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    )
]
TOOL_STRINGS = "\n".join(
    [f"{tool.name}: {tool.description}" for tool in TOOLS])
TOOL_NAMES = ", ".join([tool.name for tool in TOOLS])

PREFIX = """Please answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Please use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Please begin!

Question: {input}
Thought: {agent_scratchpad}"""

FORMAT_INSTRUCTIONS = FORMAT_INSTRUCTIONS.format(tool_names=TOOL_NAMES)


# Search agent biaed on zeroshot
@timer
def search_agent(question: str, temperature: float = None, model: str = SEARCH_MODEL) -> str:
    """
    Calls OpenAI API and searches the web to find the best answer to a question.
    """
    # Write prompt and create zero-shot agent
    prompt = ZeroShotAgent.create_prompt(
        TOOLS,
        prefix=PREFIX,
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

    response = agent_executor({'input': question})
    pretty_response = prettify_agent_response(response)

    return pretty_response
