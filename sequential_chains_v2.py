import warnings
warnings.filterwarnings('ignore')
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai

from langchain.schema.runnable import RunnableMap
from typing import List

from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.pydantic_v1 import BaseModel, Field

class Point(BaseModel):
    """Information contained in a Point."""
    id:int = Field(...,description="sequence number of point")
    argument: str = Field(..., description="argument made in the point")
    explaination: str = Field(..., description="brief explaination of the argument")
    substantiation:str = Field(...,description="supporting example or fact or report or judgement")
 

class Points(BaseModel):
    """List of points explaining given questions."""

    points: List[Point]

class intro_conclusion(BaseModel):
    """Relevant Introduction and conclusion  to question asked"""
    intro: str =Field(...,description="Introduction to answer")
    conc: str =Field(...,description="conclusion to answer")

intro_conc_parser = PydanticOutputParser(pydantic_object=intro_conclusion)
parser = PydanticOutputParser(pydantic_object=Points)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = GoogleGenerativeAI(model="gemini-pro")

def runnable_map(question):
  

  template = """
            You are a topper of UPSC civil services examination.
            You are rational , objective and well aware about current happenings in India and around world.
            You can provide strong arguments supported by evidence for any question based on context given and your knowledge. 
            context: {context}
            input: {question}
            answer: Generate atleast 5 points along with simple explaintaion,
                    and  [example] from current affairs or [fact] given by Economic survey/budget/Govt institution,  
                    or [judgement] by supreme court or highcourt or [report] by renowned committe / organisation  etc.
                    Output as per following instruction:
                    {format_instructions}
             
            """

  
  prompt = PromptTemplate(template=template, input_variables=['question','format_instructions'], output_parser=parser).partial(format_instructions=parser.get_format_instructions())
  
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
  vectordb=Chroma(embedding_function=embeddings, persist_directory="chroma_db")
  retriever = vectordb.as_retriever()

  # initialize the RunnableMap class
  chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x : x["question"],
    }) | prompt | llm | parser
# invoke the RunnableMap
  
  response = chain.invoke({"question": question})
  return response

class Subquestions(BaseModel):
    """subquestions expected directly or indirectly"""
    question:str
    type:bool =Field(..., description="True for directly asked , False for indirectly asked ")

class SubquestionsList(BaseModel):
    """subquestions expected directly or indirectly"""
    subquestions: List[Subquestions] = Field(..., description="List of subquestions expected to write")

# Set up a parser
subquestions_parser = PydanticOutputParser(pydantic_object=SubquestionsList)

template="""Identify upto 3 subquestions directly or indirectly expected to answer 
            for the question {question}
            Rearrange the questions to ensure flow and store type  as True for directly expected and False for indirectly expected.
            Output only in JSON as per following instruction:

            {format_instructions}
    """


prompt = PromptTemplate(template=template, input_variables=['question','format_instructions'], output_parser=subquestions_parser).partial(format_instructions=subquestions_parser.get_format_instructions())
    


# llm_chain = LLMChain(prompt=prompt, llm=llm)


question="""'Refugees should not be turned back to the country where they would face persecution or human right 
violation.' Examine the statement with reference to the ethical dimension being violated by the nation 
claiming to be democratic with open society."""
# subquestions=llm_chain.predict_and_parse(question=question)

chain=prompt | llm | subquestions_parser
subquestions=chain.invoke({"question": question})

answer=""
for subquestion in subquestions.subquestions:
    answer=str(answer)+"\n"
    answer=str(answer)+"_"*120 +"\n"
    answer=str(answer)+subquestion.question +"\n"
    answer=str(answer)+"_"*120 +"\n"

    response=runnable_map(subquestion.question)

    
    for point in response.points:
        answer=str(answer)+str(point.id)+". "+ point.argument +": "+ point.explaination+"\n"
        answer=str(answer)+"         "+point.substantiation+"\n"
    


format_instructions=intro_conc_parser.get_format_instructions()
prompt = PromptTemplate.from_template("""  write relevant and appealing introduction within 30 words and conclusion within 30 words
                                       based on the question: {question} and body of the answer : {answer}
                                      
                                        {format_instructions}
                                        

                         
           """)


  


chain = (
    {
        "question": itemgetter("question") ,
        "answer": itemgetter("answer"),
        "format_instructions": itemgetter("format_instructions"),
    }
    | prompt
    | llm
    | intro_conc_parser
)
response = chain.invoke({'question':question,'answer':answer,'format_instructions':format_instructions})
print("*"*60 +"final answer"+"*"*60)
print("\t"+response.intro)
print()
print(answer)
print()
print("\t"+response.conc)


# answer=""
# for subquestion in subquestions.subquestions:
#     answer=str(answer)+"\n"
#     answer=str(answer)+"_"*120 +"\n"
#     answer=str(answer)+subquestion.question +"\n"
#     answer=str(answer)+"_"*120 +"\n"

#     answer=runnable_map(subquestion.question)

    
#     for point in answer.points:
#         answer=str(answer)+str(point.id)+". "+ point.argument +": "+ point.explaination+"\n"
#         answer=str(answer)+"         "+point.substantiation+"\n"

# print(answer)

# prompt=PromptTemplate("""Rewrite the answer with relevant and appealing introduction and conclusion based on the body of the answer given context:
            
#             {answer}
                         
#            """,input_variables=['answer'])

# final_answer=(prompt | llm).invoke({'answer':answer})

# print("*"*60 +"final answer"+"*"*60)
# print(final_answer)
