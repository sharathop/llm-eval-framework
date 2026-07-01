from transformers import pipeline

# Load model once when file is imported
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small"
)

def evaluate_nli(context: str, llm_response: str) -> dict:
    
    # Combine context and response for NLI model
    input_text = f"{context} [SEP] {llm_response}"
    
    # Get prediction
    result = nli_model(input_text)
    
    label = result[0]["label"].lower()
    score = round(result[0]["score"], 4)
    
    # Map NLI label to verdict
    if label == "entailment":
        verdict = "Faithful"
    elif label == "contradiction":
        verdict = "Hallucinated"
    else:
        verdict = "Unverifiable"
    
    return {
        "label": label,
        "score": score,
        "verdict": verdict
    }

if __name__ == "__main__":

    # Should catch contradiction
    result1 = evaluate_nli(
        context="Photosynthesis is the process by which plants convert sunlight into glucose.",
        llm_response="Plants use moonlight to produce glucose through photosynthesis."
    )
    print("Moonlight test:", result1)

    # Should be faithful
    result2 = evaluate_nli(
        context="Photosynthesis is the process by which plants convert sunlight into glucose.",
        llm_response="Plants use sunlight to produce glucose through photosynthesis."
    )
    print("Sunlight test:", result2)