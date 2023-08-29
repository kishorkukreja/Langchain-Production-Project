from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host='redis', port=6379, db=0)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    conversation: List[Message]



@app.get("/service2/{conversation_id}")
async def get_conversation(conversation_id: str):
    logger.info(f"GET Retrieving initial id {conversation_id}")
    existing_conversation_json = r.get(conversation_id)
    if existing_conversation_json:
        logger.info(f"GET Connection Found {conversation_id}")
        existing_conversation = json.loads(existing_conversation_json)
        return existing_conversation
    else:
        logger.info(f"GET Connection Not Found {conversation_id}")
        return {"error": "Conversation not found"}



@app.post("/service2/{conversation_id}")
async def service2(conversation_id: str, conversation: Conversation):
    logger.info(f"POST Sending Conversation with ID {conversation_id} to OpenAI")
    existing_conversation_json = r.get(conversation_id)
    if existing_conversation_json:
        logger.info(f"POST Connection Found {conversation_id}")
        existing_conversation = json.loads(existing_conversation_json)
    else:
        logger.info(f"POST Connection Not Found {conversation_id}")
        existing_conversation = {"conversation": [{"role": "system", "content": "You are a helpful assistant."}]}

    existing_conversation["conversation"].append(conversation.dict()["conversation"][-1])

    response = requests.post(f"http://service3:80/service3/{conversation_id}", json=existing_conversation)
    response.raise_for_status()
    assistant_message = response.json()["reply"]

    existing_conversation["conversation"].append({"role": "assistant", "content": assistant_message})
    logger.info(f"POST Connection New ID {conversation_id}")
    r.set(conversation_id, json.dumps(existing_conversation))
    
    return existing_conversation


