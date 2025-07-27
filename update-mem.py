from typing import Dict, List, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
model = genai.Client(api_key=key)
history: List[Dict] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket): 
        await websocket.send_text(message)
        if message.startswith("user:"):
            pass
            
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

# Work on the vector database part

class History:
    def __init__(self):
        self.history = []
        self.context = []
        self.vectorDB = None
    
    def _cosine_sim_gem(self, text1, text2):

        result1 = [
            np.array(e.values) for e in model.models.embed_content(
                model="gemini-embedding-001",
                contents=[text1], 
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
        ]

        result2 = [
            np.array(e.values) for e in model.models.embed_content(
                model="gemini-embedding-001",
                contents=[text2], 
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
        ]

        A = np.array(result1).reshape(1, -1)
        B = np.array(result2.embeddings).reshape(1, -1)

        res = cosine_similarity(A, B)

        return res[0][0]

    def add_to_history(self, event: dict):
        now_time = datetime.now()
        if event.get("user"):
            self.history.append({"role": "user", "timestamp": now_time, "data": event})
        else:
            self.history.append({"role": "AI", "timestamp": now_time, "data": event})

        self.update_context(event)

    def update_context(self, event):
        other_event = None
        for i in range(len(self.history) - 2, -1, -1):
            curr_event = self.history[i]
            if curr_event["role"] == "user":
                similarity = self._cosine_sim_gem(self.history[i]["data"]["message"], event["message"])


manager = ConnectionManager()


@app.websocket("/ws")
async def send_message(websocket: WebSocket):
    await manager.connect(websocket)
    chat = model.chats.create(model="gemini-2.5-flash")
    while True:
        try:
            user_prompt = await manager.send_personal_message()
            if user_prompt and user_prompt["user_id"]:
                response = chat.send_message(message=user_prompt)
                await manager.send_personal_message(response.text)
        except errors.APIError(code=500):
            await manager.send_personal_message("Model has disconnected")
        except WebSocketDisconnect:
            await manager.send_personal_message("Client has disconnected")
        finally:
            await manager.disconnect(websocket)
