# Groundwork Coffee Co. RAG Assistant

A mini Retrieval-Augmented Generation (RAG) project built with LlamaIndex and OpenAI.

This project simulates a real-world business use case where an AI assistant answers questions based on internal company documents instead of relying only on general model knowledge.

## Project Overview

Groundwork Coffee Co. is a fictional community coffee shop with internal documents describing:

- menu offerings
- company history
- business hours
- loyalty program
- wholesale and catering services
- seasonal specials

The assistant loads these documents, creates vector embeddings, stores them in an in-memory vector index, and answers user questions using semantic retrieval.

## Features

- Document ingestion with `SimpleDirectoryReader`
- Semantic search with vector embeddings
- Retrieval-Augmented Generation (RAG)
- Source document inspection with similarity scores
- Failure case testing for hallucination analysis
- Reflection on retrieval vs generation limitations

## Tech Stack

- Python
- OpenAI API
- LlamaIndex
- python-dotenv

## Project Structure

```text
assignments_06/
├── warmup_06.py
├── project_06.py
└── README.md

resources/
└── groundwork_docs/
    ├── faq.txt
    ├── menu.txt
    ├── our_story.txt
    ├── seasonal_specials.txt
    └── wholesale_catering.txt
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run the Project

Execute:

```bash
python assignments_06/project_06.py
```

## Example Questions

The assistant can answer questions such as:

- What are Groundwork's hours on weekends?
- Do you offer any dairy-free milk options?
- How does the loyalty program work?
- How did Groundwork Coffee get started?
- Do you offer catering or wholesale orders?

## Failure Testing

The project includes a failure test to demonstrate a common RAG limitation.

Example:

```text
Which Groundwork location has the fastest Wi-Fi?
```

The documents mention Wi-Fi availability but do not contain speed information.

This demonstrates that even with retrieval, an LLM may still make unsupported assumptions.

## Key Learnings

This project demonstrates:

- the difference between keyword retrieval and semantic retrieval
- how frameworks like LlamaIndex reduce implementation complexity
- why retrieval quality matters
- why correct retrieval does not guarantee correct reasoning
- why AI-generated responses still require verification