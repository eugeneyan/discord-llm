# discord-llm

Code for [Experimenting with LLMs to Research, Reflect, and Plan](https://eugeneyan.com/writing/llm-experiments/). Disclaimer: The code is disorganized and hacky, and relies largely on LangChain's abstractions. May be useful as a reference, but **not** as learning material.

If you want to try this, update the `.env` file with your own keys. Most functionality, such as summarizing urls, sql queries on `/data/books.db`, and search should work right out of the box. For Q&A, you'll need to add your own custom indices.

## Discord functionality
- Summarized and ELI5 urls
- Run basic SQL via a chain or agent
- Run a search query via google custom search
- Q&A on custom indices (Note: You need to add your own indices)
