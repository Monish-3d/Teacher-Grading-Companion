import streamlit as st
from app import load_pdf, chunk_text, build_or_load_vectorstores, hybrid_evaluate

# Load and prepare data once
@st.cache_resource
def init_system():
    pdf_path = "subject_book.pdf"   # Make sure this file exists in your folder
    st.info("📖 Loading subject book...")
    raw_text = load_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    st.success("✅ Book loaded and chunked.")

    st.info("📦 Initializing vectorstores...")
    semantic_store, keyword_store = build_or_load_vectorstores(chunks)
    st.success("✅ Vectorstores ready.")
    return semantic_store, keyword_store


semantic_store, keyword_store = init_system()

# Streamlit UI config
st.set_page_config(page_title="Automatic Answer Grader", page_icon="📊", layout="centered")

st.title("📊 Teacher Grading Companion ")
st.markdown("This app uses a **RAG model** to grade student answers with reference to the textbook.")

# Inputs
question = st.text_area("✍️ Enter the Question:", height=120)
student_answer = st.text_area("📝 Enter the Student's Answer:", height=180)

# Button action
if st.button("🚀 Grade Answer"):
    if not question.strip() or not student_answer.strip():
        st.warning("⚠️ Please provide both a question and an answer.")
    else:
        with st.spinner("Evaluating... Please wait."):
            result = hybrid_evaluate(question, student_answer, semantic_store, keyword_store)

        st.subheader("✅ Evaluation Result")
        st.write(f"**LLM Score:** {result['llm_score']}/10")
        st.write(f"**Similarity Score:** {result['similarity_score']}/10")
        st.write(f"**Keyword Score:** {result['keyword_score']}/10")
        st.write(f"**Final Score:** {result['final_score']}/10")
        st.write(f"**Accuracy:** {result['accuracy']}%")
        st.markdown(f"**💡 Feedback:** {result['feedback']}")
