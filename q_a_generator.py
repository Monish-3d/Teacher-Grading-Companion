from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

class MCQ(BaseModel):
    question : str
    options : List[str]
    answer : str

class MCQSet(BaseModel):
    questions: List[MCQ]

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

struct_model = model.with_structured_output(MCQSet)

template = PromptTemplate(template="""You are a Software Engineering Professor.
                          Generate a set of High quality MCQ questions with 4 options from the given topic.
                          topic - {topic}
                          difficulty - {difficulty}
                          number of questions - {no_of_questions}""",
                          input_variables=['topic' ,'difficulty' , 'no_of_questions'])


prompt = template.invoke({'topic' : 'software development lifecycle (SDLC)',
                          'difficulty' : 'hard' ,
                          'no_of_questions' : 4})

output = struct_model.invoke(prompt)

print(output)