from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict
from _types import ContextGraph

if TYPE_CHECKING:
    from update_mem import History

class Storage:
    def __init__(self, history: 'History', persistence_dir: str = "./memory_storage"):
        self.history: 'History' = history
        self.persistence_dir: Path = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
    
    def _save_to_disk(self) -> None:
        """Persist memory to disk for recovery"""
        try:
            with open(self.persistence_dir / "history.pkl", "wb") as f:
                pickle.dump(self.history.history, f)
            
            with open(self.persistence_dir / "context_graph.pkl", "wb") as f:
                pickle.dump(self.history.context_nodes, f)
            
            metadata: Dict[str, Any] = {
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
    
    def _load_from_disk(self) -> None:
        """Load persisted memory from disk"""
        try:
            history_path: Path = self.persistence_dir / "history.pkl"
            if history_path.exists():
                with open(history_path, "rb") as f:
                    loaded_history: Any = pickle.load(f)

                    if isinstance(loaded_history, list):
                        self.history.history = loaded_history
            
            graph_path: Path = self.persistence_dir / "context_graph.pkl"
            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    loaded_context: Any = pickle.load(f)

                    if isinstance(loaded_context, ContextGraph):
                        self.history.context_nodes = pickle.load(f)
            
            metadata_path: Path = self.persistence_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata: Dict[str, Any] = json.load(f)
                    self.history._user_event_cnt = metadata.get("user_event_cnt", 0)
                    self.history.root_nodes = set(metadata.get("root_nodes", []))
                    
            print(f"Loaded {len(self.history.history)} messages from disk")  
        except Exception as e:
            print(f"Failed to load memory from disk: {e}")