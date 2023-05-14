"""
Configurations
"""
# Defaults
DEFAULT_MODEL = 'gpt-3.5-turbo'

# Config for summarization
TOKENIZER_DICT = {'gpt-3.5-turbo': 'cl100k_base',
                  'gpt-4': 'cl100k_base'}
SUMMARY_MAX_TOKENS_DICT = {'gpt-3.5-turbo': 3800,
                           'gpt-4': 7000}

SUMMARY_MODEL = DEFAULT_MODEL
SUMMARY_TOKENIZER = TOKENIZER_DICT[SUMMARY_MODEL]
SUMMARY_MAX_TOKENS = SUMMARY_MAX_TOKENS_DICT[SUMMARY_MODEL]

# Config for search
SEARCH_MODEL = DEFAULT_MODEL

# Config for SQL
SQL_MODEL = DEFAULT_MODEL

# Config for Q&A
QA_MODEL = DEFAULT_MODEL
PINECONE_ENV = 'us-west4-gcp'
PINECONE_INDEX_NAME_EY = 'ask-ey'
PINECONE_INDEX_NAME_BOARD = 'board'
EMBEDDING_MODEL = 'text-embedding-ada-002'
