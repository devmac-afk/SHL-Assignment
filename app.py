"""
app.py — Simple FastAPI backend for the SHL RAG Recommender.

Two endpoints:
  GET  /health  — check if the server is ready
  POST /chat    — send messages and get a recommendation
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import answer_question, init_rag

load_dotenv()

# ── Shared state ──────────────────────────────
retriever = None
llm = None
ready = False


# ── Startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm, ready
    print("Starting up — loading RAG system...")
    retriever, llm = init_rag()
    ready = True
    print("RAG system ready.")
    yield
    print("Shutting down.")


# ── FastAPI app ───────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[dict]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────

@app.get("/health")
def health():
    if ready:
        return {"status": "ok"}
    return {"status": "loading"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not ready:
        raise HTTPException(status_code=503, detail="Still loading, please wait.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty.")

    # The last message must be from the user
    last = request.messages[-1]
    if last.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    # Build chat history (everything except the last user message)
    chat_history = []
    for msg in request.messages[:-1]:
        role = "human" if msg.role == "user" else "ai"
        chat_history.append((role, msg.content))

    question = last.content

    reply, recommendations, end_of_conversation = answer_question(
        question, chat_history, retriever, llm
    )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )


# ── Run directly ──────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
