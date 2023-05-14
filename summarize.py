"""
Module for summarizing text.
"""
import re
from typing import List

import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.text_splitter import TokenTextSplitter

from config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL, SUMMARY_TOKENIZER
from logger import logger
from utils import timer

ENC = tiktoken.encoding_for_model(SUMMARY_MODEL)
TEXT_SPLITTER = TokenTextSplitter(encoding_name=SUMMARY_TOKENIZER)


# Count the number of tokens in text
def num_tokens(text: str) -> int:
    """
    Count the number of tokens in text.
    """
    enc = tiktoken.encoding_for_model(SUMMARY_MODEL)
    return len(enc.encode(text))


# Get text from url
def get_text_from_url(url: str) -> str:
    """
    Get text from url.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = re.sub(r'\n+', '\n', soup.get_text())  # Remove consecutive newlines

    # Trim text to 1800 tokens
    trimmed_text = ENC.decode((ENC.encode(text))[:SUMMARY_MAX_TOKENS])

    logger.info(
        f'{num_tokens(trimmed_text)}/{num_tokens(text)} tokens from {url}')
    return trimmed_text


# Get docs from text
def get_docs_from_text(text: str) -> list:
    """
    Get docs from text.
    """
    texts = TEXT_SPLITTER.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    logger.info(f'Created {len(docs):,} out of {len(texts):,} total docs')
    return docs


# Remove empty lines from text
def remove_empty_lines(text: str) -> str:
    """
    Remove empty lines from text.
    """
    return '\n'.join([line for line in text.splitlines() if line.strip()])


# Calls OpenAI API and returns summary of text
def summarize(docs: List[str], temperature: float, model: str) -> str:
    """
    Calls OpenAI API and returns summary of text.
    """
    # Write prompt
    system_msg = """You are a teacher who summarizes documents into easily digestible bullet points."""
    human_msg = """Summarize the following text in bullet points: 
    
    {text}

    Concise summary in bullet points:"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_msg),
        HumanMessagePromptTemplate.from_template(human_msg)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    logger.info(f'Prompt: {prompt}, temperature: {temperature}')

    # Create LLM and call API
    llm = OpenAI(temperature=temperature, model_name=model)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    response = chain.run(docs)
    logger.info(
        f'Results received: {response} ({num_tokens(response)} tokens), temperature: {temperature}')

    return response


# Calls OpenAI API and explains the text like the user is a five-year old
def eli5(docs: List[str], temperature: float, model: str) -> str:
    """
    Calls OpenAI API and returns explaination for a five year old
    """
    # Write prompt
    system_msg = """You are a teacher who explains documents to a five-year old."""
    human_msg = """Explain the following text to a five-year old: 
    
    {text}

    Concise explanation:"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_msg),
        HumanMessagePromptTemplate.from_template(human_msg)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    logger.info(f'Prompt: {prompt}, temperature: {temperature}')

    # Create LLM and call API
    llm = OpenAI(temperature=temperature, model_name=model)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    response = chain.run(docs)
    logger.info(
        f'Results received: {response} ({num_tokens(response)} tokens), temperature: {temperature}')

    return response


# Summarize text from url
@timer
def summarize_url(url: str, temperature: float = None, model: str = SUMMARY_MODEL) -> str:
    """
    Calls OpenAI API and returns summary of text.
    """
    logger.info(
        f'summarize: {url} (temperature: {temperature}, model: {model})')
    # Get text from url
    text = get_text_from_url(url)
    docs = get_docs_from_text(text)
    response = summarize(docs, temperature, model)
    pretty_response = remove_empty_lines(response)

    return pretty_response


# Explain like I'm five from url
@timer
def eli5_url(url: str, temperature: float = None, model: str = SUMMARY_MODEL) -> str:
    """
    Calls OpenAI API and explains the text like the user is a five-year old.
    """
    logger.info(f'eli5: {url} (temperature: {temperature}, model: {model})')
    # Get text from url
    text = get_text_from_url(url)
    docs = get_docs_from_text(text)
    response = eli5(docs, temperature, model)
    pretty_response = remove_empty_lines(response)

    return pretty_response
