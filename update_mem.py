from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
import numpy as np
import pytz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from _types import EventData, ListHistory, TimedConfigType
from vectorDB import ChromaClient

load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
model = genai.Client(api_key=key)

class HistoryNode:
    def __init__(self, event_id, event_data: EventData):
        self.id = event_id
        self.data = event_data
        self.children = []
        self.parent = None

class History:
    def __init__(self):
        self.history: ListHistory = []
        self.context_nodes: Dict[int, Union[HistoryNode, ListHistory]] = {}
        self.vectorDB = ChromaClient()
        self.timebased_memory = TimeBased()
        self.dynamic_memory = DynamicSimilarity()
    
    async def initialize(self):
        await self.vectorDB.initialize()

    async def add_to_history(self, event: dict):
        
        self.history.append(event)
        node_id = len(self.history) - 1
        new_node = HistoryNode(node_id, event)
        self.context_nodes[node_id] = new_node
        await self.vectorDB.add_event(new_node)
        self._update_context(new_node)

    async def _update_context(self, event: HistoryNode):

        if event.data.role != "user":
            return

        results = await self.vectorDB.query(event.data.message)
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
        
        best_parent_event = self.context_nodes[best_parent_id]
        new_similarity = await self.timebased_memory.apply(initial_similarity, 
                                                           best_parent_event, config={
                                                                "name": "FIXED",
                                                                "event": best_parent_event
                                                            })
    
    async def _quick_update_context(self, config: TimedConfigType):
        last: HistoryNode = self.history[-1]
        if last.data.role == "AI":
            last = self.history[-2]
    
            
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(eastern)
        if isinstance(config.type, list):
            results = await self.vectorDB.query_by_time(last.data.message, 
                                            duration=config.type, 
                                            now_time=now_time,
                                            explicit=True)
        else:
            results = await self.vectorDB.query_by_time(last.data.message,
                                                        duration=config.type,
                                                        now_time=now_time,
                                                        explicit=False) 
        
        return results
        
    

class LinkingStrategy(ABC):
    @abstractmethod
    async def apply(self, **kwargs):
        raise NotImplementedError


class TimeBased(LinkingStrategy):

    def __init__(self, tune = 0.0005, client: ChromaClient = None):
        self.client = client
        self.tune = tune
        self.dynamictime = []

    async def apply(self, *, initial_similarity: float, event: HistoryNode, parent_event: HistoryNode) -> float:
        time_difference = (event.data["timestamp"] - parent_event.data["timestamp"]).total_seconds()

        decay_factor = 1 / (1 + self.tune * time_difference)

        final_score = initial_similarity * decay_factor

        return final_score
            
            
            
class DynamicSimilarity(LinkingStrategy):
    def __init__(self, base_threshold=0.7, window_size=5, tune = 0.01):
        self.tune = tune
        self.base_threshold = base_threshold
        self.window_size = window_size

    async def apply(self, *, history: list, vector_db: ChromaClient) -> float:
        if len(history) < self.window_size:
            return self.base_threshold

       
        recent_messages_ids = [
             str(event["id"]) for event in history[-self.window_size:] if event.data.role == "user"
        ]

        if not embeddings or len(embeddings) < 2:
            return self.base_threshold
        
        embeddings = await vector_db.get_event(ids=recent_messages_ids)

        similarity_matrix = cosine_similarity(embeddings)
        
        iu = np.triu_indices(len(similarity_matrix), k=1)
        avg_similarity = np.mean(similarity_matrix[iu])

        adjusted_threshold = self.base_threshold - (avg_similarity * self.tune)
        
        return np.clip(adjusted_threshold, 0.65, 0.85)

        
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.history = History()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket): 
        await websocket.send_text(message)
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(eastern)
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
