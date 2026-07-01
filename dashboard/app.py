import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="LLM Hallucination Detector", layout="wide")

st.title("🔍 LLM Evaluation & Hallucination Detection Framework")

API_URL = "http://localhost:8000"

# --- Tabs ---
tab1, tab2 = st.tabs(["Evaluate New Response", "History"])

# ---------------- TAB 1: Evaluate ----------------
with tab1:
    st.subheader("Evaluate an LLM Response")

    context = st.text_area("Context (Ground Truth)", height=100)
    question = st.text_input("Question")
    llm_response = st.text_area("LLM Response", height=100)

    if st.button("Evaluate"):
        if not context.strip() or not question.strip() or not llm_response.strip():
            st.error("All fields are required.")
        else:
            with st.spinner("Running evaluation..."):
                response = requests.post(f"{API_URL}/evaluate", json={
                    "context": context,
                    "question": question,
                    "llm_response": llm_response
                })

            if response.status_code == 200:
                result = response.json()

                verdict = result["final_verdict"]
                if verdict == "Hallucinated":
                    st.error(f"**Verdict: {verdict}**")
                elif verdict == "Faithful":
                    st.success(f"**Verdict: {verdict}**")
                else:
                    st.warning(f"**Verdict: {verdict}**")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cosine (Relevance)", result["cosine"]["score"], result["cosine"]["verdict"])
                col2.metric("BERTScore (Faithfulness)", result["bert_score"]["score"], result["bert_score"]["verdict"])
                col3.metric("NLI", result["nli"]["score"], result["nli"]["verdict"])
                col4.metric("Fluency", "-", result["fluency"]["verdict"])

                st.json(result)
            else:
                st.error(f"Error: {response.json()['detail']}")

# ---------------- TAB 2: History ----------------
with tab2:
    st.subheader("Past Evaluations")

    if st.button("Refresh History"):
        st.rerun()

    response = requests.get(f"{API_URL}/history")

    if response.status_code == 200:
        data = response.json()
        if data["total"] == 0:
            st.info("No evaluations yet.")
        else:
            df = pd.DataFrame(data["evaluations"])
            df = df[["id", "question", "llm_response", "final_verdict", "created_at"]]
            st.dataframe(df, use_container_width=True)

            st.subheader("Verdict Distribution")
            verdict_counts = df["final_verdict"].value_counts()
            st.bar_chart(verdict_counts)
    else:
        st.error("Could not fetch history.")