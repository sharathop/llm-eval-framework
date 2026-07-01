from src.evaluators.cosine_evaluator import evaluate_cosine
from src.evaluators.fluency_evaluator import evaluate_fluency
from src.evaluators.bert_score_evaluator import evaluate_bert_score
from src.evaluators.nli_evaluator import evaluate_nli

def evaluate_all(context: str, question: str, llm_response: str) -> dict:

    # Run all 4 evaluators
    cosine_result = evaluate_cosine(question, llm_response)
    fluency_result = evaluate_fluency(llm_response)
    bert_result = evaluate_bert_score(context, llm_response)
    nli_result = evaluate_nli(context, llm_response)

    # Final verdict logic
    if nli_result["verdict"] == "Hallucinated":
        final_verdict = "Hallucinated"
    elif nli_result["verdict"] == "Faithful" and bert_result["score"] >= 0.70:
        final_verdict = "Faithful"
    elif cosine_result["verdict"] == "Irrelevant" and len(llm_response.split()) > 5:
        final_verdict = "Irrelevant"
    else:
        final_verdict = "Unverifiable"

    return {
        "final_verdict": final_verdict,
        "cosine": cosine_result,
        "fluency": fluency_result,
        "bert_score": bert_result,
        "nli": nli_result
    }

if __name__ == "__main__":
    result = evaluate_all(
        context="Photosynthesis is the process by which plants convert sunlight into glucose.",
        question="How do plants make food?",
        llm_response="Plants use moonlight to produce glucose through photosynthesis."
    )
    
    import json
    print(json.dumps(result, indent=2))