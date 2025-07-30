from abc import ABC, abstractmethod
from functools import lru_cache
import heapq
from typing import Any, Dict, List, Set, Union
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
from _types import ContextGraph, EventData, HistoryNode, TimedConfigType
from vectorDB import ChromaClient
import spacy


load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
model = genai.Client(api_key=key)
mspacy_medium = spacy.load('en_core_web_md')
mspacy_large = spacy.load('en_core_web_lg')


class History:
    def __init__(self):
        self.history: List[HistoryNode] = []
        self.context_nodes: ContextGraph = ContextGraph(context_graph={})
        self._user_event_cnt = 0
        self.top_events: List[HistoryNode] = []
        self.vectorDB = ChromaClient()
        self.timebased_memory = TimeBased()
        self.dynamic_memory = DynamicSimilarity()
    
    async def initialize(self):
        await self.vectorDB.initialize()
    
    def get_last_event(self):
        if len(self.history) <= 0:
            return None
        return self.history[-1]

    async def add_to_history(self, event: EventData):
        
        self.history.append(event)
        if event.role.lower() == "user":
            self._user_event_cnt += 1
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
        new_similarity = await self.timebased_memory.apply(initial_similarity, event, best_parent_event)
        

    async def _quick_check_context(self, config: TimedConfigType):
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


    def dfs_search(self, start_node: HistoryNode):

        visited = set()

        def dfs(current_node_id: int):
            visited.add(current_node_id)
            current_node = self.context_nodes.context_graph.get(current_node_id)
            if not current_node:
                return
            
            for child_node in current_node.children:
                if child_node.id not in visited:
                    dfs(child_node.id)

        dfs(start_node.id)
        return len(visited) - 1

    def find_candidates(self) -> Set[HistoryNode]:

        candidates: Set[HistoryNode] = set()
        
        for node in self.context_nodes.context_graph.values():
            if node.sparent is None:
                candidates.add(node)
                
        return candidates
    
    async def update_core_memory(self):

        if self._user_event_cnt == 0:
            return
        
        candidates = list(self.find_candidates())
        scores = []
        heapq.heapify(scores)
        for node in candidates:
            weight = self.dfs_search(node)
            raw_heaviness = (weight / self._user_event_cnt)
            eastern = pytz.timezone('US/Eastern')
            now_time = datetime.now(eastern)
            days = (now_time - node.data.timestamp).total_seconds() / (24 * 3600)
            age_factor = np.log(days + 1)
            heaviness_score = raw_heaviness * age_factor

            if len(scores) < 50:
                heapq.heappush(scores, (heaviness_score, node.id))
            else:
                heapq.heappushpop(scores, (heaviness_score, node.id))
        
        self.top_events = [node_id for _, node_id in scores]


    async def _quick_update_context(self, config: TimedConfigType):

        results = await self._quick_check_context(config=config)
        last_event = self.get_last_event()


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
        event_obj = EventData(role=role, timestamp=now_time, message=message)

        self.history.add_to_history(event_obj)            
        
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
