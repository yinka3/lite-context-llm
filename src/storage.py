from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from update_mem import History

class Storage:
    def __init__(self, history: 'History', persistence_dir: str = "./memory_storage"):
        self.history = history
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
    
    def _save_to_disk(self):
        """Persist memory to disk for recovery"""
        try:
            with open(self.persistence_dir / "history.pkl", "wb") as f:
                pickle.dump(self.history.history, f)  # Just the list
            
            with open(self.persistence_dir / "context_graph.pkl", "wb") as f:
                pickle.dump(self.history.context_nodes, f)
            
            metadata = {
                "user_event_cnt": self.history._user_event_cnt,
                "root_nodes": list(self.history.root_nodes),
                "last_save": datetime.now().isoformat(),
                "total_messages": len(self.history.history)
            }
            with open(self.persistence_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            print(f"Saved {len(self.history.history)} messages to disk")
                
        except Exception as e:
            print(f"Failed to save memory to disk: {e}")
    
    def _load_from_disk(self):
        """Load persisted memory from disk"""
        try:
            history_path = self.persistence_dir / "history.pkl"
            if history_path.exists():
                with open(history_path, "rb") as f:
                    loaded_history = pickle.load(f)
                    self.history.history = loaded_history  # Set the list
            
            graph_path = self.persistence_dir / "context_graph.pkl"
            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    self.history.context_nodes = pickle.load(f)
            
            metadata_path = self.persistence_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.history._user_event_cnt = metadata.get("user_event_cnt", 0)
                    self.history.root_nodes = set(metadata.get("root_nodes", []))
                    
            print(f"Loaded {len(self.history.history)} messages from disk")
            
            if len(self.history.history) > 0:
                self._rebuild_vector_db()
            
        except Exception as e:
            print(f"Failed to load memory from disk: {e}")
    
    def _rebuild_vector_db(self):
        """Rebuild the vector database from loaded history"""
        try:
            print("Rebuilding vector database...")
            for node in self.history.history:
                self.history.vectorDB.add_event(node)
            print(f"Rebuilt vector DB with {len(self.history.history)} messages")
        except Exception as e:
            print(f"Failed to rebuild vector DB: {e}")