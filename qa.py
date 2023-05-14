"""
Module for Q&A on a vector index
"""
import os

import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from config import (EMBEDDING_MODEL, PINECONE_ENV, PINECONE_INDEX_NAME_BOARD,
                    PINECONE_INDEX_NAME_EY, QA_MODEL)
from logger import logger
from utils import prettify_qa_response, timer

# Load env variables
load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize Pinecone index for ask-ey
pinecone.init(api_key=os.getenv('PINECONE_API_KEY_EY'),
              environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME_EY)
logger.info(
    f'Stats for {PINECONE_INDEX_NAME_EY}: {index.describe_index_stats()}')
store_ey = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME_EY, embedding=embeddings)

# Initialize Pinecone index for board
pinecone.init(api_key=os.getenv('PINECONE_API_KEY_BOARD'),
              environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME_BOARD)
logger.info(
    f'Stats for {PINECONE_INDEX_NAME_BOARD}: {index.describe_index_stats()}')
store_board = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME_BOARD, embedding=embeddings)


@timer
def qa_ey(question: str, temperature: float = None, model: str = QA_MODEL) -> str:

    llm = ChatOpenAI(temperature=temperature, model_name=model)
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type='stuff',
                                                        retriever=store_ey.as_retriever(),
                                                        return_source_documents=True)

    response = chain({'question': question})
    pretty_response = prettify_qa_response(response)

    return pretty_response


@timer
def qa_board(question: str, temperature: float = None, model: str = QA_MODEL) -> str:

    llm = ChatOpenAI(temperature=temperature, model_name=model)
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type='stuff',
                                                        retriever=store_board.as_retriever(),
                                                        return_source_documents=True)

    response = chain({'question': question})
    pretty_response = prettify_qa_response(response)

    return pretty_response
