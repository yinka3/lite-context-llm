from uuid import UUID, uuid4
import chromadb
import os
from chromadb.utils import embedding_functions
from typing import List, Union
from _types import HistoryNode
from datetime import datetime, timedelta
import logging


class ChromaClient:

    def __init__(self, db_id: UUID = uuid4()):
        self.db_id = db_id
        self.client = chromadb.PersistentClient(
            settings=chromadb.config.Settings(
                anonymized_telemetry=False
            )
        )

        self.embed_func_doc = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_DOCUMENT")

        self.embed_func_query = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_QUERY")

        self.collection = None

    
    def initialize(self):
        try:
            self.collection = self.client.get_or_create_collection(name="LLM_Memory", embedding_function=self.embed_func_doc)

            logging.info(f"Initialized ChromaDB with {self.collection.count} messages")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}")
            return
    
    def get_event(self, ids: List[str] | str):

        if self.collection is None:
            raise TypeError("client not initialized.")

        if not ids:
            return []
        
        results = self.collection.get(ids=ids, include=["documents"])
        return results.get('documents', [])
    
    def add_event(self, event: HistoryNode):
        if self.collection is None:
            raise TypeError("client not initialized.")
        
        try:
            document_text = event.data.message

            metadata = {
                "timestamp": event.data.timestamp.isoformat(),
                "int_time": int(event.data.timestamp.timestamp()),
                "role": event.data.role,
                "node_id": event.id
                
            }

            self.collection.add(ids=[str(event.id)], 
                                    metadatas=[metadata], 
                                    documents=[document_text])

            return True
        except Exception as e:
            logging.info(f"Count not add {event.id}: {e}")
            return False
        
    def query(self, query: str, n_results: int = 10):
        if self.collection is None:
            raise TypeError("client not initialized.")
        
        try:

            return self.collection.query(query_texts=[query],
                                         n_results=n_results, include=["metadatas", "documents", "distances"])
        except Exception as e:
            logging.error(f"Query failed for '{query}': {e}")
            return None
        

    def query_by_time(self, 
                            query: str, 
                            duration: Union[list, int], 
                            now_time: datetime,
                            explicit: bool = False, n_results: int = 10):
        if self.collection is None:
            raise TypeError("client not initialized.")

        
        try:
            if explicit and isinstance(duration, list):
                    time_delta = timedelta(
                        days=duration[0]*365 + duration[1]*30 + duration[2],
                        hours=duration[3] if len(duration) > 3 else 0
                    )
            else:
                time_delta = timedelta(days=float(duration))

            
            start_time = now_time - time_delta

            return self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where={
                        "$and": [
                            {"int_time": {"$gte": int(start_time.timestamp())}},
                            {"int_time": {"$lte": int(now_time.timestamp())}}
                        ]
                    },
                    include=["metadatas", "documents", "distances"]
                )
        
        except Exception as e:
            logging.error(f"Time-based query failed: {e}")
            return {}
    
    def query_by_date(self, query: str, target_date: datetime, window: int = 0.5, n_results: int = 10):
        if self.collection is None:
            raise TypeError("client not initialized.")
        
        try:
            start = target_date - timedelta(days=float(window))
            end = target_date + timedelta(days=float(window))
            return self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"$and": [
                            {"int_time": {"$gte": int(start.timestamp())}},
                            {"int_time": {"$lte": int(end.timestamp())}}
                        ]},
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            logging.error(f"Time-based query failed: {e}")
            return {}
    
