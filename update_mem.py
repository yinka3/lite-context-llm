from abc import ABC, abstractmethod
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
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import heapq
from vectorDB import ChromaClient

load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
model = genai.Client(api_key=key)

class HistoryNode:
    def __init__(self, event_id, event_data):
        self.id = event_id
        self.data = event_data
        self.children = []
        self.parent = None

class History:
    def __init__(self):
        self.history = []
        self.context_nodes = {}
        self.vectorDB = ChromaClient()


    def add_to_history(self, event: dict):
        
        self.history.append(event)
        node_id = len(self.history) - 1
        new_node = HistoryNode(node_id, event)
        self.context_nodes[node_id] = new_node
        self.vectorDB.add_event(new_node)
        self._update_context(new_node)

    def _update_context(self, event: HistoryNode):

        if event.data.get("role") != "user":
            return

        results = self.vectorDB.query(event.data["message"])
        best_parent_id = None
        initial_similarity = 0

        for i, node_id in enumerate(results['ids'][0]):
            if node_id != str(event.id):
                best_parent_id = int(node_id)
                initial_similarity = 1 - results['distances'][0][i]
                break
        

        if best_parent_id is None:
            print("Could not find a suitable parent, creating a new root node")
            return
        
        
    

class LinkingStrategy(ABC):
    @abstractmethod
    def apply(self, **kwargs):
        raise NotImplementedError


class TimeBased(LinkingStrategy):

    def __init__(self, tune = 0.0005):
        self.tune = tune
    
    def apply(self, *, initial_similarity: float, event: HistoryNode, parent_event: HistoryNode) -> float:
        time_difference = (event.data["timestamp"] - parent_event.data["timestamp"]).total_seconds()


        decay_factor = 1 / (1 + self.tune * time_difference)

        final_score = initial_similarity * decay_factor

        return final_score
    

class DynamicSimilarity(LinkingStrategy):
    def __init__(self, base_threshold=0.7, window_size=5, tune = 0.01):
        self.tune = tune
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.vectorizer = TfidfVectorizer()

    def apply(self, *, history: list) -> float:
        if len(history) < self.window_size:
            return self.base_threshold

        # Get the text from the last few user messages
        recent_messages = [
            event["message"] for event in history[-self.window_size:] if event.get("role") == "user"
        ]

        if len(recent_messages) < 2:
            return self.base_threshold

        tfidf_matrix = self.vectorizer.fit_transform(recent_messages)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])

        adjusted_threshold = self.base_threshold - (avg_similarity * self.tune)
        
        return max(adjusted_threshold, 0.6)

        
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.history = History()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket): 
        await websocket.send_text(message)
        now_time = datetime.now()
        role = "user" if message.startswith("user:") else "AI"
        event = {"role": role, "timestamp": now_time, "message": message}
        self.history.add_to_history(event)            
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

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
