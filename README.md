# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for website.

The bot answers questions using only content retrieved from the website.

It uses:

- OpenAI (LLM + embeddings)
- ChromaDB (Vector db)
- FastAPI
- Playwright / requests (For website indexing)

Rules the bot uses are really easy to change. Currently bot works in finnish.

How it works:

1. indexer.py:
   - Reads the website sitemap.
   - Fetches and parses pages.
   - Splits content into chunks.
   - Creates embeddings.
   - Stores them into ChromaDB.

2. /chat endpoint
   - Retrieves relevant chunks from ChromaDB.
   - Builds a context block.
   - Sends context and the question to OpenAI.
   - Returns the response.
  
There is no installation guide for this, but if you want to see this in work, its on my project website https://bittipuisto.fi (Bottom right corner).

Chatbot runs on private VPS server 24/7 using systemd.
