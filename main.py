"""
Bot for discord server that utilizes OpenAI's api for commands.
"""
import argparse
import os
from sqlite3 import OperationalError

import interactions
from dotenv import load_dotenv

from config import DEFAULT_MODEL
from logger import logger
from qa import qa_board, qa_ey
from search import search_agent
from sql import sql_agent, sql_chain
from summarize import eli5_url, summarize_url

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='prod')
args = parser.parse_args()
logger.info(f'Arguments: {args.__dict__}')

# Discord arguments
MAX_INITIAL_MESSAGE_LENGTH = 1900
MAX_MESSAGE_LENGTH = 2000

# Load environment variables
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD_ID = os.getenv('DISCORD_GUILD_ID')
CMD_PREFIX = ''

if args.env == 'dev':
    TOKEN = os.getenv('DISCORD_TOKEN_DEV')
    CMD_PREFIX = 'dev-'

bot = interactions.Client(TOKEN)
logger.info(f'Bot initialized: {bot.__dict__}')

# Define reusable options
OPTIONS_TEMPERATURE = interactions.Option(name='temperature', description='Lower values = more focused responses, higher values = more random', required=False,
                                          type=interactions.OptionType.NUMBER, min_value=0.0, max_value=2.0)
OPTIONS_MODEL = interactions.Option(name='model', description='Model to use', required=False,
                                    type=interactions.OptionType.STRING,
                                    choices=[interactions.Choice(name='gpt-3.5', value='gpt-3.5-turbo'),
                                             interactions.Choice(name='gpt-4', value='gpt-4')])
OPTIONS_SHOW_SOURCE = interactions.Option(name='show_source', description='Show snippets of source content', required=False,
                                          type=interactions.OptionType.BOOLEAN,
                                          choices=[interactions.Choice(name='yes', value=True),
                                                   interactions.Choice(name='no', value=False)])


@bot.command(name=f'{CMD_PREFIX}hello', description='Says hello without hitting any APIs. Used for health checks.', scope=GUILD_ID)
async def _hello(ctx: interactions.CommandContext):
    await ctx.send(f'Hello {ctx.author.mention}! How are you?')


@bot.command(name=f'{CMD_PREFIX}summarize', description='Summarizes a URL in bullet points', scope=GUILD_ID,
             options=[interactions.Option(name='url', description='URL to summarize', required=True, type=interactions.OptionType.STRING),
                      OPTIONS_TEMPERATURE, OPTIONS_MODEL])
async def _summarize(ctx: interactions.CommandContext, url: str, temperature: float = None, model: str = DEFAULT_MODEL):
    logger.info(f'Summarize: {url}, Temp: {temperature}, Model: {model}')
    await ctx.defer()
    summary, time = summarize_url(url, temperature, model)
    summary += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
    await ctx.send(f'Here is the summary of {url}:\n\n{summary[:MAX_INITIAL_MESSAGE_LENGTH]}')
    for i in range(MAX_INITIAL_MESSAGE_LENGTH, len(summary), MAX_MESSAGE_LENGTH):
        await ctx.send(f'{summary[i:i+MAX_MESSAGE_LENGTH]}')


@ bot.command(name=f'{CMD_PREFIX}eli5', description='Explains a URL to a five-year old', scope=GUILD_ID,
              options=[interactions.Option(name='url', description='URL to explain', required=True, type=interactions.OptionType.STRING),
                       OPTIONS_TEMPERATURE, OPTIONS_MODEL])
async def _eli5(ctx: interactions.CommandContext, url: str, temperature: float = None, model: str = DEFAULT_MODEL):
    logger.info(f'ELI5: {url}, Temp: {temperature}, Model: {model}')
    await ctx.defer()
    explanation, time = eli5_url(url, temperature, model)
    explanation += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
    await ctx.send(f'Here is the explanation of {url}:\n\n{explanation[:MAX_INITIAL_MESSAGE_LENGTH]}')
    for i in range(MAX_INITIAL_MESSAGE_LENGTH, len(explanation), MAX_MESSAGE_LENGTH):
        await ctx.send(f'{explanation[i:i+MAX_MESSAGE_LENGTH]}')


@bot.command(name=f'{CMD_PREFIX}search', description='Searches the internet for a query', scope=GUILD_ID,
             options=[interactions.Option(name='query', description='Query to search for', required=True, type=interactions.OptionType.STRING),
                      OPTIONS_TEMPERATURE, OPTIONS_MODEL])
async def _search_agent(ctx: interactions.CommandContext, query: str, temperature: float = None, model: str = DEFAULT_MODEL):
    logger.info(f'Search: {query}, Temp: {temperature}, Model: {model}')
    await ctx.defer()
    try:
        result, time = search_agent(query, temperature, model)
        result += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
        await ctx.send(f'{result[:MAX_INITIAL_MESSAGE_LENGTH]}')
        for i in range(MAX_INITIAL_MESSAGE_LENGTH, len(result), MAX_MESSAGE_LENGTH):
            await ctx.send(f'{result[i:i+MAX_MESSAGE_LENGTH]}')
    except ValueError as e:
        await ctx.send(f'Error: {e}. Please try again.')


@bot.command(name=f'{CMD_PREFIX}table', description='Describes the books table.', scope=GUILD_ID)
async def _table(ctx: interactions.CommandContext):
    await ctx.send(f'The books table has the following columns: id, title, author, language, average rating, ratings count, and text reviews count.')


@bot.command(name=f'{CMD_PREFIX}sql', description='Queries a database', scope=GUILD_ID,
             options=[interactions.Option(name='query', description='Query to search for', required=True, type=interactions.OptionType.STRING),
                      OPTIONS_TEMPERATURE, OPTIONS_MODEL])
async def _sql_chain(ctx: interactions.CommandContext, query: str, temperature: float = None, model: str = DEFAULT_MODEL):
    logger.info(f'SQL-chain: {query}, Temp: {temperature}, Model: {model}')
    await ctx.defer()
    try:
        result, time = sql_chain(query, temperature, model)
        result += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
        await ctx.send(f'{result[:MAX_INITIAL_MESSAGE_LENGTH]}')
        for i in range(MAX_INITIAL_MESSAGE_LENGTH, len(result), MAX_MESSAGE_LENGTH):
            await ctx.send(f'{result[i:i+MAX_MESSAGE_LENGTH]}')
    except OperationalError as e:
        await ctx.send(f'Error: {e}. Please try again.')


@bot.command(name=f'{CMD_PREFIX}sql-agent', description='Queries a database', scope=GUILD_ID,
             options=[interactions.Option(name='query', description='Query to search for', required=True, type=interactions.OptionType.STRING),
                      OPTIONS_TEMPERATURE, OPTIONS_MODEL])
async def _sql_agent(ctx: interactions.CommandContext, query: str, temperature: float = None, model: str = DEFAULT_MODEL):
    logger.info(f'SQL-agent: {query}, Temp: {temperature}, Model: {model}')
    await ctx.defer()
    try:
        result, time = sql_agent(query, temperature, model)
        result += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
        await ctx.send(f'{result[:MAX_INITIAL_MESSAGE_LENGTH]}')
        for i in range(MAX_INITIAL_MESSAGE_LENGTH, len(result), MAX_MESSAGE_LENGTH):
            await ctx.send(f'{result[i:i+MAX_MESSAGE_LENGTH]}')
    except ValueError as e:
        await ctx.send(f'Error: {e}. Please try again.')


@bot.command(name=f'{CMD_PREFIX}ask-ey', description='Asks eugeneyan.com a question', scope=GUILD_ID,
             options=[interactions.Option(name='question', description='Question to ask', required=True, type=interactions.OptionType.STRING),
                      OPTIONS_TEMPERATURE, OPTIONS_MODEL, OPTIONS_SHOW_SOURCE])
async def _ask_ey(ctx: interactions.CommandContext, question: str, temperature: float = None, model: str = DEFAULT_MODEL, show_source: bool = False):
    logger.info(
        f'Ask ey: {question}, Temp: {temperature}, Model: {model}, Show source: {show_source}')
    await ctx.defer()
    # The first element is the answer, the rest are sources
    result_list, time = qa_ey(question, temperature, model)

    result = result_list[0]
    result += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
    await ctx.send(f'{result[:MAX_MESSAGE_LENGTH]}')

    if show_source:
        # Send sources as individual messages
        for source in result_list[1:]:
            await ctx.send(f'{source[:MAX_MESSAGE_LENGTH]}')


@bot.command(name=f'{CMD_PREFIX}board', description='Asks board of advisors a question', scope=GUILD_ID,
             options=[interactions.Option(name='question', description='Question to ask', required=True, type=interactions.OptionType.STRING),
                      OPTIONS_TEMPERATURE, OPTIONS_MODEL, OPTIONS_SHOW_SOURCE])
async def _ask_board(ctx: interactions.CommandContext, question: str, temperature: float = None, model: str = DEFAULT_MODEL, show_source: bool = False):
    logger.info(
        f'Ask board: {question}, Temp: {temperature}, Model: {model}, Show source: {show_source}')
    await ctx.defer()
    # The first element is the answer, the rest are sources
    result_list, time = qa_board(question, temperature, model)

    result = result_list[0]
    result += f'\n\n `Temp: {temperature}, Model: {model}, Time: {time:.2f}s`'
    await ctx.send(f'{result[:MAX_MESSAGE_LENGTH]}')

    if show_source:
        # Send sources as individual messages
        for source in result_list[1:]:
            await ctx.send(f'{source[:MAX_MESSAGE_LENGTH]}')


if __name__ == '__main__':
    bot.start()
