"""
Utility functions for the project.
"""
import re
from functools import wraps
from time import perf_counter
from typing import Callable


# Timer decorator
def timer(func: Callable) -> Callable:
    """
    Timer decorator.
    """
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = perf_counter()
        value = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        print(f'Finished {func.__name__!r} in {run_time:.2f} secs')
        return value, run_time

    return wrapper_timer


# Prettify langchain agent response
def prettify_agent_response(response: dict, input_key: str = 'input', output_key: str = 'output') -> str:
    """
    Pretty print the response from the agent.
    """
    pretty_result = ''

    pretty_result += f'**Input:** {response[input_key]}\n\n'

    for step in response['intermediate_steps']:
        action = step[0]
        result = step[1]
        # pretty_result += f'**Tool**: {action.tool} | **Input**: "{action.tool_input}"\n'
        pretty_result += f'**Thought:** {action.log}\n'
        pretty_result += f'**Observation:** _{result}_\n\n'

    pretty_result += f'\n**Output:** {response[output_key]}'

    return pretty_result


# Prettify langchain agent response
def prettify_chain_response(response: dict, input_key: str = 'query', output_key: str = 'result') -> str:
    """
    Pretty print the response from the chain.
    """
    pretty_result = ''

    pretty_result += f'**Input:** {response[input_key]}\n\n'

    if 'intermediate_steps' in response:
        for i, step in enumerate(response['intermediate_steps']):
            pretty_result += f'**Step {i}:** {step}\n\n'

    pretty_result += f'\n**Output:** {response[output_key]}'

    return pretty_result


# Wrap urls in <> to prevent discord from embedding them
def wrap_urls(text):
    return re.sub(r'(https?://\S+)', r'<\1>', text)


# Prettify langchain Q&A chain response
def prettify_qa_response(response: dict, question_key: str = 'question', answer_key: str = 'answer') -> str:
    """
    Pretty print the response from the Q&A chain.
    """
    result_list = []
    pretty_qa = ''

    pretty_qa += f'**Question:** {response[question_key]}\n\n'
    pretty_qa += f'**Answer:** {response[answer_key]}\n'
    pretty_qa += f'**Sources:** {wrap_urls(response["sources"])}'
    result_list.append(pretty_qa)

    for doc in response['source_documents']:
        pretty_source = ''
        pretty_source += f'**Source:** {wrap_urls(doc.page_content)}\n\n'
        pretty_source += f'**URL:** {wrap_urls(doc.metadata["source"])}'
        result_list.append(pretty_source)

    return result_list
