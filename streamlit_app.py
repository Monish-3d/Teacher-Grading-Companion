import streamlit as st
from app import load_pdf, chunk_text, build_or_load_vectorstores, chain

# ---------------------------
# Wrapper to use your new chain
# ---------------------------
def run_hybrid_evaluation(question, answer, subject, semantic_store, keyword_store):
    result = chain.invoke({
        "question": question,
        "answer": answer,
        "subject": subject,
        "semantic_store": semantic_store,
        "keyword_store": keyword_store
    })
    return result


# ---------------------------
# Available Subjects
# ---------------------------
subject_books = {
    "Software Engineering": "subject_book.pdf",
    "Object Oriented Programming": "OOP_book.pdf",
}

# ---------------------------
# Cache Loading + Vectorstores
# ---------------------------
@st.cache_resource(show_spinner=False)
def init_subject(subject):
    pdf_path = subject_books[subject]
    st.info(f"📖 Loading {subject} textbook...")
    raw_text = load_pdf(pdf_path)

    chunks = chunk_text(raw_text)
    st.success(f"✅ {subject} book loaded and chunked.")

    st.info("📦 Building vectorstores (semantic + keyword)...")
    semantic_store, keyword_store = build_or_load_vectorstores(chunks, subject)
    st.success(f"✅ Vectorstores for {subject} ready.")

    return semantic_store, keyword_store


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Hybrid Grader", page_icon="📘", layout="centered")

st.title("📊 Teacher Grading Companion")
st.markdown("Grade student answers intelligently using a **Hybrid RAG Model** with textbook references.")

subject = st.sidebar.selectbox("📚 Select Subject", options=list(subject_books.keys()))
semantic_store, keyword_store = init_subject(subject)

st.subheader("🧾 Student Answer Evaluation")
question = st.text_area("✍️ Enter the Question:", height=120)
student_answer = st.text_area("📝 Enter the Student's Answer:", height=180)

if st.button("🚀 Evaluate Answer"):
    if not question.strip() or not student_answer.strip():
        st.warning("⚠️ Please provide both the question and the student's answer.")
    else:
        with st.spinner(f"Evaluating answer for {subject}..."):
            result = run_hybrid_evaluation(
                question, student_answer, subject,
                semantic_store, keyword_store
            )

        st.subheader("✅ Evaluation Result")
        st.metric("LLM Score", f"{result['llm_score']}/10")
        st.metric("Similarity Score", f"{result['similarity_score']}/10")
        st.metric("Keyword Score", f"{result['keyword_score']}/10")

        st.markdown(f"### **Final Score: {result['final_score']}/10**")
        st.progress(min(result["final_score"] / 10, 1.0))

        st.markdown(f"**Accuracy:** {result['accuracy']}%")
        st.markdown(f"💡 **Feedback:** {result['feedback']}")

st.markdown("---")
