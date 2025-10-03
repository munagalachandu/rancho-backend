# app.py
import os
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chatbot import load_faiss_index, get_qa_chain
from datetime import datetime, timezone
import motor.motor_asyncio
from uuid import uuid4
from typing import Optional
# MongoDB connection setup (adjust URI for your setup)
MONGODB_URI = "mongodb+srv://chandureddy8325_db_user:chandu123@cluster0.f3hnlmm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "chat_db"
COLLECTION_NAME = "messages"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # add your frontend URLs here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = load_faiss_index()
qa_chain = get_qa_chain(vectorstore)

class ChatRequest(BaseModel):
    user_id: str
    message: str

def start_of_day(dt: datetime):
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_id = chat_request.user_id
    message = chat_request.message
    today_start = start_of_day(datetime.utcnow())

    # Find today's chat or create new
    today_chat = await collection.find_one(
        {
            "user_id": user_id,
            "timestamp": {"$gte": today_start}
        },
        sort=[("timestamp", 1)]
    )

    if today_chat:
        chat_id = today_chat["chat_id"]
    else:
        chat_id = str(uuid4())

    # Store user message
    await collection.insert_one({
        "user_id": user_id,
        "chat_id": chat_id,
        "role": "user",
        "content": message,
        "timestamp": datetime.utcnow()
    })

    # Generate bot reply
    response = qa_chain({"query": message})
    bot_reply = response["result"]

    # Store bot message
    await collection.insert_one({
        "user_id": user_id,
        "chat_id": chat_id,
        "role": "assistant",
        "content": bot_reply,
        "timestamp": datetime.utcnow()
    })

    # Get updated chat messages for today
    cursor = collection.find({"user_id": user_id, "chat_id": chat_id}).sort("timestamp", 1)
    history = []
    async for doc in cursor:
        history.append({
            "role": doc["role"],
            "content": doc["content"],
            "timestamp": doc["timestamp"].isoformat()
        })

    return {
        "reply": bot_reply,
        "chat_id": chat_id,
        "history": history
    }

@app.get("/current_chat")
async def get_current_chat(user_id: str = Query(...)):
    today_start = start_of_day(datetime.utcnow())
    today_chat = await collection.find_one(
        {
            "user_id": user_id,
            "timestamp": {"$gte": today_start}
        }
    )

    if not today_chat:
        return {"user_id": user_id, "chat": [], "message": "No chat today."}

    chat_id = today_chat["chat_id"]

    cursor = collection.find({"user_id": user_id, "chat_id": chat_id}).sort("timestamp", 1)
    chat_messages = []
    async for doc in cursor:
        chat_messages.append({
            "role": doc["role"],
            "content": doc["content"],
            "timestamp": doc["timestamp"].isoformat()
        })

    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "chat": chat_messages
    }

@app.get("/chat/history")
async def get_chat_history(user_id: str):
    today_start = start_of_day(datetime.utcnow())

    chat_ids = await collection.distinct("chat_id", {
        "user_id": user_id,
        "timestamp": {"$lt": today_start}
    })

    history_summary = []
    for cid in chat_ids:
        first_msg = await collection.find_one(
            {"user_id": user_id, "chat_id": cid},
            sort=[("timestamp", 1)]
        )
        if first_msg:
            history_summary.append({
                "chat_id": cid,
                "first_message": first_msg["content"],
                "date": first_msg["timestamp"].date().isoformat()
            })

    history_summary.sort(key=lambda x: x["date"], reverse=True)

    return {"user_id": user_id, "previous_chats": history_summary}

@app.get("/chat/full_history")
async def get_full_chat_history(user_id: str, chat_id: str):
    cursor = collection.find({
        "user_id": user_id,
        "chat_id": chat_id
    }).sort("timestamp", 1)

    messages = []
    async for doc in cursor:
        messages.append({
            "role": doc["role"],
            "content": doc["content"],
            "timestamp": doc["timestamp"].isoformat()
        })
    return {"user_id": user_id, "chat_id": chat_id, "messages": messages}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
