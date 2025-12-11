import os
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
#-------------------pinecone setup----------------------------------------

index_name = "subject-books-embeddings"
pinecone_api = os.getenv("PINECONE_API_KEY") 
pc = Pinecone(api_key = pinecone_api)

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"no index with name {index_name} found")
else:
    print("index found:" , index_name)

embeddings = GoogleGenerativeAIEmbeddings(model = 'models/gemini-embedding-001')
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index , embedding = embeddings)
#----------------------------building prompt-----------------------------------

class MCQ(BaseModel):
    question : str
    options : List[str]
    answer : str

class MCQSet(BaseModel):
    questions: List[MCQ]

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
struct_model = model.with_structured_output(MCQSet)

template = PromptTemplate(template="""You are a Software Engineering Professor.
                          Generate a set of High quality MCQ questions with 4 options from the given topic with given context from the subject book.
                          generate with your understanding ONLY if you feel the given context is insufficient for the required no of questions.
                          topic - {topic}
                          difficulty - {difficulty}
                          number of questions - {no_of_questions}
                          Reference context - {context}""",
                          input_variables=['topic' ,'difficulty' , 'no_of_questions' , 'context'])
#----------------------function---------------------------------

def generate_mcqs(topic:str , difficulty:str , no_of_questions : int):
    retriever = vector_store.as_retriever(search_type ='similarity', search_kwargs= {'k':10})
    retrieved_docs = retriever.invoke(topic)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = template.invoke({'topic' : topic,
                          'difficulty' : difficulty ,
                          'no_of_questions' : no_of_questions ,
                          'context' : context_text})
    
    output = struct_model.invoke(prompt)
    return output
#------------------main function-----------------------------------------

if __name__ == "__main__":

    topic = 'software development Life Cycle (SDLC)'
    res = generate_mcqs(topic=topic , difficulty='hard' , no_of_questions = 2)

    print(res)
