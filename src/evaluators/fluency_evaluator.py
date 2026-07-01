import re

def evaluate_fluency(llm_response: str)  -> dict:
    issues = []

    if len(llm_response.split()) <5:
        issues.append("Response too short")

    if len(llm_response.split()) >500:
        issues.append("Response too long")

    if not llm_response[0].isupper():
        issues.append("Does not start with capital letter")

    if llm_response[-1] not in [".", "?", "!"]:
        issues.append("Does not end with proper punctuation")

    words = llm_response.lower().split()
    for i in range(len(words)-1):
        if words[i] == words[i+1]:
            issues.append("contains repeated consecutive words")
            break
    

    if len(issues) == 0:
        verdict = "Fluent"
    elif len(issues) <= 2:
        verdict = "Partially Fluent"
    else:
        verdict = "Not Fluent"
    
    return {
        "issues": issues,
        "verdict": verdict
    }


# if __name__ == "__main__":


#     result1 = evaluate_fluency("The Eiffel Tower is located in Paris, France.")
#     print("Good response:", result1)

#     # Bad response
#     result2 = evaluate_fluency("yes")
#     print("Bad response:", result2)

#     # Repeated words
#     result3 = evaluate_fluency("The the Eiffel Tower is in Paris.")
#     print("Repeated words:", result3)