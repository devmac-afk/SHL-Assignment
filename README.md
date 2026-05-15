# SHL Conversational Assessment Recommender

FastAPI + Streamlit RAG application built for the SHL AI Intern take-home assignment.

## What It Does

- Exposes `GET /health` and `POST /chat`
- Accepts full stateless conversation history on every `/chat` call
- Clarifies vague requests before recommending assessments
- Supports follow-up refinements by rewriting the latest user turn with chat history
- Refuses prompt-injection and clearly out-of-scope requests
- Returns grounded SHL catalog recommendations with names, URLs, and test types

## Assignment Schema

`POST /chat` returns:

```json
{
  "reply": "Here are 3 assessments that fit your role.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "Knowledge & Skills Test"
    }
  ],
  "end_of_conversation": true
}
```

`recommendations` is empty while the agent is clarifying, refusing, or comparing.

## Project Structure

```text
app.py
rag.py
streamlit_app.py
render.yaml
requirements.txt
data/shl_catalog.json
vectorstore/
tests/test_assignment_contract.py
```

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set `GOOGLE_API_KEY` in `.env`.

3. Start the API:

```bash
uvicorn app:app --reload --port 8000
```

4. Start the UI:

```bash
streamlit run streamlit_app.py
```

Set `BACKEND_URL` in Streamlit secrets or environment variables for deployed use.

## Tests

Run the deterministic contract checks with:

```bash
python -m unittest discover -s tests -v
```

The suite covers:

- API schema compliance
- vague-query clarification
- prompt-injection refusal
- out-of-scope refusal
- comparison detection
- scope filtering for non-test catalog artifacts

## Deployment

`render.yaml` deploys the FastAPI app directly with:

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

The application builds the retriever during startup and serves readiness on `/health`.
