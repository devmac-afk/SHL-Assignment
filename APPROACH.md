# Approach Document

## System Design

The project is a stateless conversational recommender over the SHL catalog. The backend exposes `GET /health` and `POST /chat` through FastAPI. Every `/chat` request carries the full message history, and the service stores no per-conversation session state. The frontend is a thin Streamlit client that persists chat history in the browser session and reposts it on each turn.

The retrieval layer uses a lightweight setup:

- A simple character-based text chunker to divide document text into chunks.
- A BM25 keyword search index for retrieval.

*Note on Architecture Compromises:* The original plan was to use a hybrid retrieval approach (`sentence-transformers` + FAISS for semantic search, and `Docling`'s `HybridChunker` for structure-aware parsing). However, Render's free tier limits memory to 512MB, which caused Out-Of-Memory (OOM) crashes during deployment when loading heavy ML libraries like PyTorch. To successfully deploy the app, the heavy ML libraries and semantic search were stripped out in favor of a purely keyword-based BM25 retriever and a standard character splitter.

## Prompting And Conversation Handling

The agent has four main behaviors aligned to the brief:

1. Clarify vague requests before recommending.
2. Recommend grounded SHL assessments once enough context exists.
3. Refine recommendations when the user changes constraints.
4. Compare assessments using retrieved catalog evidence rather than model priors.

To improve multi-turn behavior, the latest user message is rewritten into a standalone retrieval query using chat history. This prevents follow-ups like “add personality tests” or “show more options” from losing earlier role context during retrieval.

The backend also has rule-based guards for:

- prompt injection attempts
- clearly out-of-scope requests
- first-turn vague prompts

These guards return a refusal or clarifying question with an empty recommendation list, preserving schema compliance.

## Catalog Scope

The assignment restricts recommendations to Individual Test Solutions. The provided dataset does not expose a clean structured field for this, so the current implementation uses a conservative text filter to exclude obvious out-of-scope items such as solutions, reports, guides, and profiler cards. This is a pragmatic safeguard, though a production version should prefer a first-class catalog field if SHL exposes one.

## Evaluation Approach

I added deterministic tests for the highest-risk requirements:

- API schema compliance including `end_of_conversation`
- vague query clarification
- prompt-injection refusal
- out-of-scope refusal
- comparison detection
- catalog scope filtering

Given more time, I would add replay-based evaluation against the public conversation traces and compute Recall@10 directly over those traces.

## What Did Not Work

- Relying only on the latest user turn for retrieval caused weak follow-up handling.
- The initial dataset ingestion was too permissive and allowed non-test artifacts.
- The original deployment file referenced build steps that did not exist in the repo.
- Deploying heavy ML dependencies (PyTorch, Docling, FAISS) on Render's 512MB free tier resulted in immediate Out-of-Memory crashes, forcing a downgrade to a pure keyword-search (BM25) approach.

## AI Tooling Disclosure

AI-assisted coding was used to speed up implementation, review the repository against the assignment brief, and draft code changes. All final logic, filters, and behavior decisions were manually checked and adjusted against the project requirements.
