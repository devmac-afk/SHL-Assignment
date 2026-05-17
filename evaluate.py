import os
import json
import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = None
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import init_rag, answer_question, get_llm

# ─────────────────────────────────────────────────────────────
# Evaluation Dataset
# ─────────────────────────────────────────────────────────────
# A small set of queries with their expected recommended test names.
# This tests Retrieval Quality (Recall).
EVALUATION_DATASET = [
    {
        "query": "Recommend an assessment for a Java developer.",
        "expected_test_names": ["Java 8 (New)"]
    },
    {
        "query": "I need a test for a customer service representative.",
        "expected_test_names": ["Customer Service Scenario"]
    },
    {
        "query": "What do you have for logical reasoning?",
        "expected_test_names": ["Verify - Logical Reasoning"]
    }
]

# ─────────────────────────────────────────────────────────────
# LLM-as-a-Judge Prompts
# ─────────────────────────────────────────────────────────────

EVAL_PROMPT = """You are an impartial evaluator for an AI RAG system.
You will be provided with:
1. The user's query.
2. The AI's response.
3. The context retrieved from the database.

You must score the AI's response on two metrics from 1 to 5.
1. Relevance (1-5): Does the response directly address the user's query?
2. Groundedness (1-5): Is the response entirely based on the provided context without hallucinating outside information?

Return your answer strictly in valid JSON format like this:
{{
    "relevance_score": 4,
    "groundedness_score": 5,
    "reasoning": "brief explanation"
}}

User Query: {query}
AI Response: {response}
Retrieved Context: {context}
"""

def run_evaluation():
    print("Initializing RAG system for evaluation...")
    retriever, llm = init_rag()
    eval_llm = get_llm()
    eval_chain = ChatPromptTemplate.from_template(EVAL_PROMPT) | eval_llm | StrOutputParser()

    total_recall = 0
    total_relevance = 0
    total_groundedness = 0
    num_queries = len(EVALUATION_DATASET)

    print("\nStarting Evaluation...")
    print("=" * 60)

    for i, data in enumerate(EVALUATION_DATASET, 1):
        query = data["query"]
        expected_tests = data["expected_test_names"]
        
        print(f"\nTest {i}/{num_queries}: {query}")
        
        # 1. Run the RAG pipeline
        # We manually retrieve the docs here just to pass to the evaluator
        retrieved_docs = retriever.invoke(query)
        context_str = "\n".join([doc.page_content for doc in retrieved_docs])
        
        reply, recommendations, _ = answer_question(query, [], retriever, llm)
        
        # 2. Measure Retrieval Quality (Recall)
        # Check if expected tests are in the recommendations list
        recommended_names = [rec["name"] for rec in recommendations]
        recall_score = 0
        for expected in expected_tests:
            if any(expected.lower() in rec_name.lower() for rec_name in recommended_names):
                recall_score = 1
                break
        total_recall += recall_score
        
        # 3. LLM-as-a-Judge for Relevance and Groundedness
        eval_result_str = eval_chain.invoke({
            "query": query,
            "response": reply,
            "context": context_str
        })
        
        # Clean markdown formatting from the response if present
        if eval_result_str.startswith("```json"):
            eval_result_str = eval_result_str.strip("```json").strip("```")
            
        try:
            eval_metrics = json.loads(eval_result_str)
            relevance = eval_metrics.get("relevance_score", 0)
            groundedness = eval_metrics.get("groundedness_score", 0)
        except json.JSONDecodeError:
            print("Failed to parse LLM judge JSON.")
            relevance = 0
            groundedness = 0
            
        total_relevance += relevance
        total_groundedness += groundedness
        
        print(f"  Retrieval Quality (Recall): {recall_score}/1")
        print(f"  Relevance Score: {relevance}/5")
        print(f"  Groundedness Score: {groundedness}/5")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS")
    print("=" * 60)
    print(f"Average Retrieval Quality (Recall): {(total_recall / num_queries) * 100:.2f}%")
    print(f"Average Relevance: {total_relevance / num_queries:.2f} / 5.0")
    print(f"Average Groundedness: {total_groundedness / num_queries:.2f} / 5.0")

if __name__ == "__main__":
    run_evaluation()
