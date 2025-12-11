import os
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
api_key = os.getenv("API_KEY")             
pinecone_api = os.getenv("PINECONE_API_KEY") 
index_name = "se-book-embeddings"

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="embedding-001",
#     google_api_key=api_key
# )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.0
)


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_text(text)



def init_pinecone():
    pc = Pinecone(api_key=pinecone_api)

    if index_name not in [i["name"] for i in pc.list_indexes()]:
        print("⚡ Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=768,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  
        )
    else:
        print("🔄 Pinecone index already exists")

    return pc.Index(index_name)



def build_or_load_vectorstores(chunks):
    index = init_pinecone()
    stats = index.describe_index_stats()

    if stats.total_vector_count == 0:
        print("⚡ Uploading chunks to Pinecone...")
        semantic_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        print("🔄 Using existing Pinecone index...")
        semantic_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )

    keyword_retriever = BM25Retriever.from_texts(chunks)

    return semantic_store, keyword_retriever


response_schemas = [
    ResponseSchema(name="score", description="Score out of 10 as an integer"),
    ResponseSchema(name="feedback", description="1–2 sentence feedback")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

EVAL_PROMPT = """
You are a software engineering professor. 
Grade the student's answer based on the context provided from the textbook.

QUESTION: {question}

REFERENCE CONTEXT:
{context}

STUDENT ANSWER:
{answer}

{format_instructions}

IMPORTANT:
- Only return an integer for "score" (0 to 10), do not write "x/10".
"""


prompt = PromptTemplate(
    input_variables=["question", "context", "answer"],
    template=EVAL_PROMPT,
    partial_variables={"format_instructions": format_instructions}
)




def hybrid_retrieve(question, semantic_retriever, keyword_retriever, k=3):
    semantic_docs = semantic_retriever.as_retriever(search_kwargs={"k": k}).get_relevant_documents(question)
    keyword_docs = keyword_retriever.get_relevant_documents(question)

    seen, combined = set(), []
    for d in semantic_docs + keyword_docs:
        if d.page_content not in seen:
            combined.append(d)
            seen.add(d.page_content)
    return combined[:k]



def hybrid_evaluate(question, answer, semantic_store, keyword_store):
    docs = hybrid_retrieve(question, semantic_store, keyword_store, k=3)
    context = "\n".join([d.page_content for d in docs])

    
    formatted_prompt = prompt.format(question=question, context=context, answer=answer)
    llm_result = llm.invoke(formatted_prompt).content
    try:
        parsed = output_parser.parse(llm_result)
        llm_score = int(parsed.get("score", 0))
        llm_feedback = parsed.get("feedback", "")
    except Exception as e:
        llm_score, llm_feedback = 0, f"LLM grading failed: {str(e)}"

    
    ref_text = context[:600]
    ref_emb = embeddings.embed_query(ref_text)
    stu_emb = embeddings.embed_query(answer)
    sim = cosine_similarity([ref_emb], [stu_emb])[0][0]
    sim_score = round(sim * 10, 2) 

   
    keywords = [w.lower() for w in question.split() if len(w) > 4]
    answer_words = answer.lower().split()
    keyword_hits = sum(1 for k in keywords if k in answer_words)
    keyword_score = round((keyword_hits / max(1, len(keywords))) * 10, 2)

   
    final_score = round(0.6*llm_score + 0.2* sim_score + 0.2*keyword_score, 2)
    accuracy = round((sim_score * 10 + keyword_score * 10) / 2, 2) 

    return {
        "llm_score": llm_score,
        "similarity_score": sim_score,
        "keyword_score": keyword_score,
        "final_score": final_score,
        "accuracy": accuracy,
        "feedback": f"{llm_feedback} (Similarity: {sim_score}/10, Keywords: {keyword_score}/10)"
}


if __name__ == "__main__":
    pdf_path = "subject_book.pdf"

    print("📖 Loading book...")
    raw_text = load_pdf(pdf_path)
    chunks = chunk_text(raw_text)

    print("📦 Building / Loading Vectorstores...")
    semantic_store, keyword_store = build_or_load_vectorstores(chunks)

    examples = [
        {
            "q": "What is the difference between functional and non-functional requirements?",
            "a": "Functional requirements describe what the system should do, while non-functional requirements describe how the system should perform, like speed or reliability."
        },
        {
            "q": "Explain the Waterfall model in software engineering.",
            "a": "The Waterfall model is a linear approach where phases like requirements, design, implementation, and testing happen sequentially."
        },
        {
            "q": "What is software testing?",
            "a": "Software testing is the process of finding errors in the system and making sure it works as expected."
        },
        {
            "q": "Define software engineering.",
            "a": "It is the application of engineering principles to design, develop, and maintain software systems."
        }
    ]

    for ex in examples:
        print("\n" + "="*60)
        print(f"📌 Question: {ex['q']}")
        print(f"✍️ Student Answer: {ex['a']}")

        result = hybrid_evaluate(ex["q"], ex["a"], semantic_store, keyword_store)

        print("\n✅ Hybrid Evaluation Result:")
        print(f"LLM Score: {result['llm_score']}/10")
        print(f"Similarity Score: {result['similarity_score']}/10")
        print(f"Keyword Score: {result['keyword_score']}/10")
        print(f"Final Score: {result['final_score']}/10")
        print(f"Accuracy: {result['accuracy']}%")
        print(f"Feedback: {result['feedback']}")

