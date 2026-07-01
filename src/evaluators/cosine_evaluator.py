from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_cosine(question: str, llm_response: str) -> dict:

    question_embedding = model.encode([question])
    response_emebedding = model.encode([llm_response])

    score = cosine_similarity(question_embedding, response_emebedding)[0][0]
    score = round(float(score), 4)
    if score >= 0.7:
        verdict = "Relevant"
    elif score >= 0.4:
        verdict = "Partially Relevant"
    else:
        verdict = "Irrelevant"
    
    return {
        "score": score,
        "verdict": verdict
    }

# if __name__ == "__main__":
#     result = evaluate_cosine(
#         question="Where is the Eiffel Tower?",
#         llm_response="Photosynthesis is how plants make food from sunlight."
#     )
#     print(result)



