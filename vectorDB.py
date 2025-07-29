from uuid import UUID, uuid4
import chromadb
import os
import logging
from chromadb.utils import embedding_functions
from chromadb.api.client import AsyncClient
from typing import Any, List, Union
from update_mem import HistoryNode
from datetime import datetime, timedelta

class ChromaClient:

    def __init__(self, db_id: UUID = uuid4()):
        self.db_id = db_id
        self.client = None

        self.embed_func_doc = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_DOCUMENT")

        self.embed_func_query = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_QUERY")

    
    async def initialize(self):
        if self.client is None:
            self.client: 'chromadb.AsyncHttpClient' = chromadb.AsyncHttpClient()
            self.collection = await self.client.get_or_create_collection(
                name="LLM_Memory")
    
    async def get_event(self, ids: List[str]):

        if not ids:
            return []
        
        results = await self.collection.get(ids=ids, include=["embeddings"])
        return results.get('embeddings', [])
    
    async def add_event(self, event: HistoryNode):
        document_text = event.data["data"]["message"]

        metadata = {
            "timestamp": str(event.data["timestamp"]),
            "role": event.data["role"]
        }

        await self.collection.add(ids=[str(event.id)], 
                            metadatas=[metadata], 
                            documents=[document_text])

    async def query(self, query: str):
        query_embedding = await self.embed_func_query([query])

        return await self.collection.query(query_embeddings=query_embedding, n_results=5)

    async def query_by_time(self, query: str, duration: Union[list, datetime], now_time: str, explicit: bool = False):

        query_embedding = await self.embed_func_query([query])
        time = None
        format_string = "%Y-%m-%d %H:%M:%S"
        now_datetime_obj = datetime.strptime(now_time, format_string)
        if explicit == True:
            time_diff = datetime(year=duration[0], month=duration[1], day=duration[2], hour=duration[3])
            time = now_datetime_obj - time_diff
        else:
            time_diff = timedelta(days=duration)
            time = now_datetime_obj - time_diff
        
        return await self.collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                where={
                    "timestamp":
                    {
                        "$lte":  now_time,
                        "$gte": time
                    }
                }
            )


            
        


