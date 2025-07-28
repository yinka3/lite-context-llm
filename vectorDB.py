from uuid import UUID, uuid4
import chromadb
import os
import logging
from chromadb.utils import embedding_functions
from typing import Any, Union
from update_mem import HistoryNode

class ChromaClient:

    def __init__(self, db_id: UUID = uuid4(), embed_func = None):
        self.db_id = db_id
        self.client = chromadb.PersistentClient(f"./chroma_db_{self.db_id}")

        self.embed_func_doc = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_DOCUMENT")

        self.embed_func_query = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["SECRET_KEY"], task_type="RETRIEVAL_QUERY")

        self.collection = self.client.get_or_create_collection(
            name="LLM_Memory",
            embedding_function=self.embed_func_doc)
    
    def add_event(self, event: HistoryNode):
        document_text = event.data["data"]["message"]

        metadata = {
            "timestamp": str(event.data["timestamp"]),
            "role": event.data["role"]
        }

        self.collection.add(ids=[str(event.id)], 
                            metadatas=[metadata], 
                            documents=[document_text])

    def query(self, query: str):
        query_embedding = self.embed_func_query([query])

        return self.collection.query(query_embeddings=query_embedding, n_results=5)


