from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
from langchain.prompts import PromptTemplate
import logging
import os
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

##with agent
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)

ROLE_CLASS_MAP = {
    "assistant": AIMessage,
    "user": HumanMessage,
    "system": SystemMessage
}

#load_dotenv(find_dotenv())
#openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key ="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"
#load_dotenv(find_dotenv())
CONNECTION_STRING = "postgresql+psycopg2://admin:admin@postgres:5432/vectordb"
COLLECTION_NAME="vectordb"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    conversation: List[Message]

embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(temperature=0)


store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

retriever = store.as_retriever()



prompt_template = """You are an AI assistant named Atlas, created to be helpful, harmless, and honest. 
You can respond to natural language requests and queries related to procurement data. 
Your role is to serve as a buyer co-pilot for an organization, retrieving data from vector stores and providing relevant information to the user.
You have the information about these columns:
COMMODITY COMMODITY_DESCRIPTION EXTENDED_DESCRIPTION QUANTITY UNIT_OF_MEASURE UNIT_OF_MEAS_DESC UNIT_PRICE ITM_TOT_AM MASTER_AGREEMENT CONTRACT_NAME PURCHASE_ORDER AWARD_DATE VENDOR_CODE LGL_NM AD_LN_1 AD_LN_2 CITY ST ZIP CTRY DATA_BUILD_DATE Spend Class Contract PO No. of Vendors Vendor Class No. of POs PO Class

For data like the sample provided, with fields including COMMODITY, COMMODITY_DESCRIPTION, EXTENDED_DESCRIPTION, QUANTITY, UNIT_OF_MEASURE, and others, I can:

-Retrieve specific data fields and values when asked, such as the UNIT_PRICE for a given COMMODITY
-Answer questions about relationships between data fields, like which PURCHASE_ORDERs correspond to a given VENDOR_CODE
-Compare and analyze data values across records, such as identifying the COMMODITY with the highest ITM_TOT_AM
-Synthesize insights from the data, like the most common VENDOR_CODEs by SPEND_CLASS
-Among other insights

Your role is to serve as an analytical assistant, leveraging the structured data to 
provide helpful responses to natural language queries that support data-driven procurement decisions and processes. My responses will be based solely on the data provided to you, and guided by the principles of being helpful, harmless, and honest.
As a Buyer Co-Pilot for our organization, you have the following information about the different commodities:
{context}

Please provide the most suitable response for the users question.
If you do not understand the question or donot have full information about the data, please say I do not understand.
Answer:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)



#search = GoogleSearchAPIWrapper()
#tools = [
#    Tool(
#        name="Search",
#        func=search.run,
#        description="useful for when you need to answer questions about current events",
#    )
#]

#prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:
#            You are an AI assistant named Atlas, created to be helpful, harmless, and honest. 
#            You can respond to natural language requests and queries related to procurement data. 
#            Your role is to serve as a buyer co-pilot for an organization, retrieving data from vector stores and providing relevant information to the user."""
#suffix = """Begin!"

#{chat_history}
#Question: {input}
#{agent_scratchpad}"""

#prompt = ZeroShotAgent.create_prompt(
#    tools,
#    prefix=prefix,
#    suffix=suffix,
#    input_variables=["input", "chat_history", "agent_scratchpad"],
#)
#memory = ConversationBufferMemory(memory_key="chat_history")


#llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
#agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
#agent_chain = AgentExecutor.from_agent_and_tools(
#    agent=agent, tools=tools, verbose=True, memory=memory
#)

prompt_template_csv = """Have a conversation with a human, answering the following questions as best you can. 
            You are an AI assistant named Atlas, created to be helpful, harmless, and honest. 
            You can respond to natural language requests and queries related to procurement data. 
            Your role is to serve as a buyer co-pilot for an organization, retrieving data from vector stores and providing relevant information to the user.
            {context}"""
#suffix = """Begin!"


prompt_csv = PromptTemplate(
    template=prompt_template_csv,input_variables=["context"]
)

system_message_prompt_csv = SystemMessagePromptTemplate(prompt=prompt_csv)

path = os.getcwd()
logger.info(path)

file_path = "./input_data.csv"
csv_memory = ConversationBufferMemory()

csv_agent = create_csv_agent(ChatOpenAI(temperature=0,
                                            model="gpt-3.5-turbo-0613"), 
                                            file_path,
                                            memory=csv_memory,
                                            verbose=True,
                                            handle_parsing_errors=True,
                                            agent_type=AgentType.OPENAI_FUNCTIONS,
                                            )


def create_messages(conversation):
    return [ROLE_CLASS_MAP[message.role](content=message.content) for message in conversation]
    
def create_messages_csv_agent(conversation):
    return [ROLE_CLASS_MAP[message.role](content=message.content) for message in conversation]

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_doc = "Source: " + doc.metadata['source']
        formatted_docs.append(formatted_doc)
    return '\n'.join(formatted_docs)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/service3/{conversation_id}")
async def service3(conversation_id: str, conversation: Conversation):

    query = conversation.conversation[-1].content

    docs = retriever.get_relevant_documents(query=query)
    docs = format_docs(docs=docs)

    prompt = system_message_prompt.format(context=docs)
    messages = [prompt] + create_messages(conversation=conversation.conversation)

    result = chat(messages)

    return {"id": conversation_id, "reply": result.content}


# @app.post("/service3/{conversation_id}")
# async def service3(conversation_id: str, conversation: Conversation):

#     query = conversation.conversation[-1].content

#     #docs = retriever.get_relevant_documents(query=query)
#     #docs = format_docs(docs=docs)

#     #prompt = system_message_prompt.format(context=docs)
#     #messages = [prompt] + create_messages(conversation=conversation.conversation)
    
#     prompt = system_message_prompt_csv.format(context="You will answer the questions")
#     messages = [prompt] + create_messages_csv_agent(conversation=conversation.conversation)
    
#     result = csv_agent.run(messages)
#     #agent_chain.run(input="How many people live in canada?")

#     return {"id": conversation_id, "reply": result}