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
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda , RunnableBranch

#-----------------------load-keys---------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")             
pinecone_api = os.getenv("PINECONE_API_KEY") 
index_name = "subject-books-embeddings"

#-----------------------embeddings---------------------------------------------

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001",
#     google_api_key=api_key
# )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#-----------------------LLM model----------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.0
)

#-----------------------load and split/chunk text---------------------------------------------

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
    return splitter.split_text(text) # return a list of string

#-----------------------inititialize vector database---------------------------------------------

def init_pinecone(index_name):
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

def build_or_load_vectorstores(chunks , subject):
    
    index = init_pinecone(index_name)

    stats = index.describe_index_stats()

    subject_exists = False

    if "namespaces" in stats and subject in stats["namespaces"]:
        ns_stats = stats["namespaces"][subject]
        if ns_stats["vector_count"] > 0:
            subject_exists = True
    
    if not subject_exists:
        print(f"⚡ Uploading {subject} chunks to Pinecone...")
        semantic_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=subject,
            metadatas = [{"subject" : subject} for _ in chunks]
        )
        print("Chunks uploaded")
    else:
        print("🔄 Using existing Pinecone index...")
        semantic_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=subject
        )

    keyword_retriever = BM25Retriever.from_texts(chunks)

    return semantic_store, keyword_retriever

#-------------------Prompt Template--------------------------------------------

response_schemas = [
    ResponseSchema(name="score", description="Score out of 10 as an integer"),
    ResponseSchema(name="feedback", description="1–2 sentence feedback")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

EVAL_PROMPT = """
You are a university professor. 
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
#---------------------------retrieval-----------------------------------------------

def hybrid_retrieve(question,subject, semantic_retriever, keyword_retriever, k=3):
    
    semantic_docs = semantic_retriever.as_retriever(
        search_type = 'similarity',
        search_kwargs={
            "k": k,
            "filter": {"subject":subject}
        }
    ).invoke(question)

    keyword_docs = keyword_retriever.invoke(question)

    seen, combined = set(), []
    for d in semantic_docs + keyword_docs:
        if d.page_content not in seen:
            combined.append(d)
            seen.add(d.page_content)
    return combined[:k]


# def hybrid_evaluate(question, answer, subject ,semantic_store, keyword_store):
#     docs = hybrid_retrieve(question, subject , semantic_store, keyword_store, k=3)
#     context = "\n".join([d.page_content for d in docs])

#     formatted_prompt = prompt.format(question=question, context=context, answer=answer)
#     llm_result = llm.invoke(formatted_prompt).content
#     try:
#         parsed = output_parser.parse(llm_result)
#         llm_score = int(parsed.get("score", 0))
#         llm_feedback = parsed.get("feedback", "")
#     except Exception as e:
#         llm_score, llm_feedback = 0, f"LLM grading failed: {str(e)}"

    
#     ref_text = context[:600]
#     ref_emb = embeddings.embed_query(ref_text)
#     stu_emb = embeddings.embed_query(answer)
#     sim = cosine_similarity([ref_emb], [stu_emb])[0][0]
#     sim_score = round(sim * 10, 2) 
   
#     keywords = [w.lower() for w in question.split() if len(w) > 4]
#     answer_words = answer.lower().split()
#     keyword_hits = sum(1 for k in keywords if k in answer_words)
#     keyword_score = round((keyword_hits / max(1, len(keywords))) * 10, 2)

#     final_score = round(0.6*llm_score + 0.2* sim_score + 0.2*keyword_score, 2)
#     accuracy = round((sim_score * 10 + keyword_score * 10) / 2, 2) 

#     return {
#         "Subject" : subject,
#         "llm_score": llm_score,
#         "similarity_score": sim_score,
#         "keyword_score": keyword_score,
#         "final_score": final_score,
#         "accuracy": accuracy,
#         "feedback": f"{llm_feedback} (Similarity: {sim_score}/10, Keywords: {keyword_score}/10)"
# }

def retrieve_context(inputs):
        question = inputs["question"]
        subject = inputs["subject"]
        semantic = inputs["semantic_store"]
        keyword = inputs["keyword_store"]
        docs = hybrid_retrieve(question, subject, semantic, keyword, k=3)
        context = "\n".join([d.page_content for d in docs])
        return {"context": context, **inputs}

def llm_grade(inputs):
    formatted_prompt = prompt.format(
        question=inputs["question"],
        context=inputs["context"],
        answer=inputs["answer"]
    )
    llm_result = llm.invoke(formatted_prompt).content
    try:
        parsed = output_parser.parse(llm_result)
        llm_score= int(parsed.get("score", 0)),
        feedback= parsed.get("feedback", "")
        
    except Exception as e:
        llm_score ,feedback = 0 , f"Parse error: {str(e)}"
    
    return {
        **inputs,
        "llm_score": llm_score,
        "feedback": feedback
    }

def postprocess(inputs):
    context, answer, question = inputs["context"], inputs["answer"], inputs["question"]

    ref_text = context[:600]
    ref_emb = embeddings.embed_query(ref_text)
    stu_emb = embeddings.embed_query(answer)
    sim = cosine_similarity([ref_emb], [stu_emb])[0][0]
    sim_score = round(sim * 10, 2)

    keywords = [w.lower() for w in question.split() if len(w) > 4]
    answer_words = answer.lower().split()
    keyword_hits = sum(1 for k in keywords if k in answer_words)
    keyword_score = round((keyword_hits / max(1, len(keywords))) * 10, 2)

    print("🔍 LLM SCORE RAW:", inputs.get("llm_score"))

    llm_score = inputs.get("llm_score", 0)
    if isinstance(llm_score, tuple):
        llm_score = llm_score[0]


    final_score = round(0.6*float(llm_score) + 0.2*sim_score + 0.2*keyword_score, 2)
    accuracy = round((sim_score * 10 + keyword_score * 10) / 2, 2)

    return {
        "Subject": inputs["subject"],
        "llm_score": llm_score,
        "similarity_score": sim_score,
        "keyword_score": keyword_score,
        "final_score": final_score,
        "accuracy": accuracy,
        "feedback": f"{inputs['feedback']} (Similarity: {sim_score}/10, Keywords: {keyword_score}/10)"
    }

chain = RunnableSequence(RunnablePassthrough() , RunnableLambda(retrieve_context), RunnableLambda(llm_grade),RunnableLambda(postprocess))

#----------------------main------------------------------------------------

if __name__ == "__main__":

    subject_pdf_map = {
        "se" : "subject_book.pdf",
        "oops" : "OOP_book.pdf"
    }

    subject = input("Enter subject (se / oops / etc): ").strip().lower()

    if subject not in subject_pdf_map:
        raise ValueError(f"Invalid subject '{subject}'. Choose from: {list(subject_pdf_map.keys())}")

    pdf_path = subject_pdf_map[subject]

    #----------------------------------------------------------------------

    # parallel_chain = RunnableParallel({
    # 'context': RunnableLambda(build_or_load_vectorstores) | RunnableLambda(hybrid_retrieve),
    # 'question': RunnablePassthrough(),
    # 'answer' : RunnablePassthrough(),
    # 'subject' : RunnablePassthrough()
    # })

    # final_chain = parallel_chain | prompt | llm | output_parser
    #----------------------------------------------------------------------

    print(f"📖 Loading {subject} book...")
    raw_text = load_pdf(pdf_path)
    chunks = chunk_text(raw_text)

    print("📦 Building / Loading Vectorstores...")
    semantic_store, keyword_store = build_or_load_vectorstores(chunks , subject)

    examples = [
        {
            "q": "What are objects in Object Oriented Programming",
            "a": "Objects are the basic run-time entities in an object-oriented system.They may represent a person, a place, a bank account, a table of data or any item that the program has to handle. They may also represent user-defined data such as vectors, time and lists. Programming problem is analyzed in terms of objects and the nature of communication between them."
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

        result = chain.invoke({
        "question": ex['q'],
        "answer": ex['a'],
        "subject": subject,
        "semantic_store": semantic_store,
        "keyword_store": keyword_store
        })

        #result = hybrid_evaluate(ex["q"], ex["a"] ,subject ,semantic_store, keyword_store)

        print("\n✅ Hybrid Evaluation Result:")
        print(f"LLM Score: {result['llm_score']}/10")
        print(f"Similarity Score: {result['similarity_score']}/10")
        print(f"Keyword Score: {result['keyword_score']}/10")
        print(f"Final Score: {result['final_score']}/10")
        print(f"Accuracy: {result['accuracy']}%")
        print(f"Feedback: {result['feedback']}")
