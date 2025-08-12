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
        self.client = chromadb.PersistentClient()

        self.embed_func_doc = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_DOCUMENT")

        self.embed_func_query = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_QUERY")

        self.collection = None
        self.memory_manager = MemoryManager()

    
    async def initialize(self):
        try:
            self.collection = self.client.get_or_create_collection(name="LLM_Memory", embedding_function=self.embed_func_doc)

            count = self.collection.count()
            self.memory_manager.message_count = count
            logging.info(f"Initialized ChromaDB with {count} messages")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}")
            return
    
    async def get_event(self, ids: List[str] | str):

        if self.collection is None:
            raise TypeError("client not initialized.")

        if not ids:
            return []
        
        results = await self.collection.get(ids=ids, include=["documents"])
        return results.get('documents', [])
    
    async def add_event(self, event: HistoryNode):
        if self.collection is None:
            raise TypeError("client not initialized.")
        
        try:
            document_text = event.data.message

            metadata = {
                "timestamp": event.data.timestamp.isoformat(),
                "int_time": int(event.data.timestamp.timestamp()),
                "role": event.data.role,
                "node_id": event.id
                # add another metadata called llm_summary, make a small llm model give it a topic phrase
            }

            self.collection.add(ids=[str(event.id)], 
                                    metadatas=[metadata], 
                                    documents=[document_text])

            self.memory_manager.update_count(added=1)

            if await self.memory_manager.should_cleanup():
                await self._perform_cleanup()

            return True
        except Exception as e:
            logging.info(f"Count not add {event.id}: {e}")
            return False
        
    async def query(self, query: str, n_results: int = 10):
        if self.collection is None:
            raise TypeError("client not initialized.")
        
        try:

            return self.collection.query(query_texts=[query],
                                         n_results=n_results, include=["metadatas", "documents", "distances"])
        except Exception as e:
            logging.error(f"Query failed for '{query}': {e}")
            return None
        

    async def query_by_time(self, 
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
    
    async def query_by_date(self, query: str, target_date: datetime, window: int = 0.5, n_results: int = 10):
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
    
    async def _perform_cleanup(self):
        try:
            being_removed = await self.memory_manager.get_cleanup_candidates(self.collection)

            if being_removed:
                self.collection.delete(ids=being_removed)
                self.memory_manager.update_count(removed=len(being_removed))
                logging.info(f"Cleaned up {len(being_removed)} old messages")
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")


class MemoryManager:

    def __init__(self, max_messages: int = 50000, cleanup_batch: int = 1000):
        self.message_count = 0
        self.max_messages = max_messages
        self.cleanup_batch = cleanup_batch
        self.cleanup_threshold = int(max_messages * 0.9)
    

    async def should_cleanup(self) -> bool:
        return self.message_count >= self.cleanup_threshold
    
    async def get_cleanup_candidates(self, collection) -> List[str]:
        try:
            results = collection.get(
                limit=self.cleanup_batch * 2,
                include=["metadatas"]
            )
            
            if not results['ids']:
                return []
            
            items = []
            for i, meta in enumerate(results["metadatas"]):
                if 'int_time' in meta:
                    items.append((results['ids'][i], meta['int_time']))

            items.sort(key=lambda x: x[1])
            
            return [item[0] for item in items[:self.cleanup_batch]]
            
        except Exception as e:
            logging.error(f"Failed to get cleanup candidates: {e}")
            return []
    
    def update_count(self, added: int = 0, removed: int = 0):
        self.message_count += added - removed
        logging.info(f"Message count: {self.message_count}/{self.max_messages} ({self.message_count/self.max_messages*100:.1f}% full)")


















            
        


