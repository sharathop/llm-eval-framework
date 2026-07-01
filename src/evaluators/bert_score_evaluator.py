from bert_score import score

def evaluate_bert_score(context: str, llm_response:str) -> dict:

    P, R, F1 = score(
        cands = [llm_response],
        refs = [context],
        lang ="en",
        verbose =False 

    )

    f1_score =round(float(F1[0]), 4)

    if f1_score >= 0.85:
        verdict = "Highly Faithful"
    elif f1_score >= 0.70:
        verdict = "Faithful"
    elif f1_score >= 0.50:
        verdict = "Partially Faithful"
    else:
        verdict = "Unfaithful"
    
    return {
        "score": f1_score,
        "verdict": verdict
    }


if __name__ == "__main__":

    # Faithful response
    result1 = evaluate_bert_score(
        context="Photosynthesis is the process by which plants convert sunlight into glucose.",
        llm_response="Plants use sunlight to produce glucose through photosynthesis."
    )
    print("Faithful:", result1)

    # Unfaithful response
    result2 = evaluate_bert_score(
        context="Photosynthesis is the process by which plants convert sunlight into glucose.",
        llm_response="Plants use moonlight to produce glucose through photosynthesis."
    )
    print("Unfaithful:", result2)