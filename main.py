from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.aggregator import evaluate_all
from src.database import init_db, save_evaluation

app = FastAPI(
    title="LLM Evaluation & Hallucination Detection Framework",
    version="1.0.0"
)
init_db()


# Define what the request should look like
class EvalRequest(BaseModel):
    context: str
    question: str
    llm_response: str

# Define what the response will look like
class EvalResponse(BaseModel):
    final_verdict: str
    cosine: dict
    fluency: dict
    bert_score: dict
    nli: dict

@app.get("/")
def home():
    return {"message": "LLM Evaluation Framework is running"}

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(request: EvalRequest):

    # Edge case — empty inputs
    if not request.context.strip():
        raise HTTPException(status_code=400, detail="Context cannot be empty")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not request.llm_response.strip():
        raise HTTPException(status_code=400, detail="LLM response cannot be empty")

    # Run evaluation
    result = evaluate_all(
        context=request.context,
        question=request.question,
        llm_response=request.llm_response
    )
    save_evaluation(request.context, request.question, request.llm_response, result)

    return result

from src.database import get_all_evaluations

@app.get("/history")
def history():
    rows = get_all_evaluations()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "context": row[1],
            "question": row[2],
            "llm_response": row[3],
            "final_verdict": row[4],
            "created_at": row[11]
        })
    
    return {"total": len(results), "evaluations": results}