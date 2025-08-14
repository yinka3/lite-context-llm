from typing import Dict, List, Optional, Set
import numpy as np
import pytz
import numpy as np
from datetime import datetime
from _types import ContextGraph, EventData, HistoryNode, TimedConfigType
from vectorDB import ChromaClient
from storage import Storage

# want to use spacy models for topic generation to group
# try:
#     import spacy
#     mspacy_medium = spacy.load('en_core_web_md')
# except Exception as e:
#     print(f"Spacy model not load")
#     mspacy_medium = None


class History:
    MAX_CAPACITY = 50000

    def __init__(self):
        self.history: List[HistoryNode] = []
        self._ephemeral_history: List[HistoryNode] = []
        self.context_nodes: ContextGraph = ContextGraph(context_graph={})
        self.root_nodes: Set[int] = set()
        self._user_event_cnt = 0
        self.top_events: List[HistoryNode] = []

        self.vectorDB = ChromaClient()
        self.storage = Storage(history=self)

        self.storage._load_from_disk()

    def initialize(self):
        self.vectorDB.initialize()
    
    def add_to_history(self, event: EventData):
        
        try:
            if event.role.lower() == "user":
                self._user_event_cnt += 1
            node_id = len(self.history)
            new_node = HistoryNode(node_id, event)
            self.context_nodes.context_graph[node_id] = new_node
            self.vectorDB.add_event(new_node)
            self.history.append(new_node)
            self._update_context(new_node)

            if len(self.history) % 5 == 0:
                self.storage._save_to_disk()
                
        except Exception as e:
            print(f"issue went wrong: {e}")
            return

    def _update_context(self, event: HistoryNode):
        if event.data.role != "user":
            return
        
        try:
            results = self.vectorDB.query(event.data.message)
            if not results or not results.get('ids') or not results['ids'][0]:
                print(f"No vector results found for event {event.id}, creating root node")
                event.sparent = None
                return
            
            best_parent_id = None
            best_score = 0
            
            for i, node_id in enumerate(results['ids'][0]):
                candidate_id = int(node_id)
                
                if candidate_id == event.id:
                    continue
                
                if candidate_id not in self.context_nodes.context_graph:
                    continue
                
                if self._would_create_cycle(event, candidate_id):
                    continue
                
                candidate_node = self.context_nodes.context_graph[candidate_id]
                similarity = 1 - results['distances'][0][i]
                
                final_score = self.apply_time_decay(
                    similarity, 
                    event.data.timestamp, 
                    candidate_node.data.timestamp
                )
                
                if final_score > best_score:
                    best_score = final_score
                    best_parent_id = candidate_id
            
            if best_parent_id is None:
                print(f"No suitable parent found for event {event.id}, creating root node")
                event.sparent = None
                return
            
            threshold = self.get_similarity_threshold(len(self.history))
            
            if best_score >= threshold:
                best_parent_event = self.context_nodes.context_graph[best_parent_id]
                event.sparent = best_parent_event
                best_parent_event.children.append(event)
                print(f"✓ Connected: Message {event.id} → Message {best_parent_id} (score: {best_score:.3f})")
            else:
                print(f"Score {best_score:.3f} below threshold {threshold:.3f}, making root node")
                event.sparent = None
                self.root_nodes.add(event.id)       
        except Exception as e:
            print(f"Critical error in context building for event {event.id}: {e}")
            
    
    def get_similarity_threshold(self, message_count: int) -> float:
        """Progressive threshold: 0.51 to 0.80 as memory fills"""
        fullness = message_count / self.MAX_CAPACITY
        return 0.51 + (0.25 * fullness)

    def apply_time_decay(self, similarity: float, current_time: datetime, candidate_time: datetime) -> float:
        """Linear decay with grace period for recent messages"""
        hours_ago = (current_time - candidate_time).total_seconds() / 3600
        
        if hours_ago < 2:
            return similarity
        
        days_ago = hours_ago / 24
        time_multiplier = max(0.4, 1 - (days_ago * 0.05))
        return similarity * time_multiplier

    def get_relevant_context(self, query: str, max_results: int = 5) -> List[Dict]:
        """Get relevant historical context for a query"""
        try:
            results = self.vectorDB.query(query, n_results=max_results)
            
            if not results or not results.get('ids') or not results['ids'][0]:
                return []
            
            relevant_contexts = []
            
            for i, node_id in enumerate(results['ids'][0]):
                node_id = int(node_id)
                
                if node_id not in self.context_nodes.context_graph:
                    continue
                
                node = self.context_nodes.context_graph[node_id]
                
                conversation_thread = self.get_conversation_story(node_id)
                
                context = {
                    'matched_message': node.data.message,
                    'role': node.data.role,
                    'timestamp': node.data.timestamp,
                    'similarity': 1 - results['distances'][0][i],
                    'conversation_thread': [
                        {
                            'role': n.data.role,
                            'message': n.data.message,
                            'timestamp': n.data.timestamp.isoformat()
                        }
                        for n in conversation_thread[-5:]
                    ]
                }
                relevant_contexts.append(context)
            
            relevant_contexts.sort(key=lambda x: x['similarity'], reverse=True)
            
            return relevant_contexts
            
        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return []
    def _would_create_cycle(self, new_child: HistoryNode, potential_parent_id: int) -> bool:

        current_id = potential_parent_id
        visited = set()
        
        while current_id is not None and current_id not in visited:
            if current_id == new_child.id:
                return True
            
            visited.add(current_id)
            current_node = self.context_nodes.context_graph.get(current_id)
            current_id = current_node.sparent.id if current_node and current_node.sparent else None
        
        return False

    def get_conversation_story(self, top_event_id: int) -> List[HistoryNode]:
        
        if top_event_id not in self.context_nodes.context_graph:
            return []
        
        path = []
        current = self.context_nodes.context_graph[top_event_id]
        
        while current is not None:
            path.append(current)
            current = current.sparent

        return list(reversed(path))

    def get_story_branches(self, root_event_id: int) -> Dict[str, List[HistoryNode]]:
        
        if root_event_id not in self.context_nodes.context_graph:
            return {}
        
        root = self.context_nodes.context_graph[root_event_id]
        branches = {}
        
        def dfs_story(node: HistoryNode, path: List[HistoryNode], branch_name: str):
            current_path = path + [node]
            
            if not node.children:
                branches[f"Branch_{branch_name}_{len(branches)}"] = current_path
            else:
                for i, child in enumerate(node.children):
                    dfs_story(child, current_path, f"{branch_name}_{i}")
        
        dfs_story(root, [], "main")
        return branches


    def _quick_check_context(self, config: TimedConfigType):
        last: HistoryNode = self.history[-1]
        if last.data.role == "AI":
            last = self.history[-2]
    
            
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(eastern)
        if isinstance(config.type, list):
            results = self.vectorDB.query_by_time(last.data.message, 
                                            duration=config.type, 
                                            now_time=now_time,
                                            explicit=True)
        else:
            results = self.vectorDB.query_by_time(last.data.message,
                                                        duration=config.type,
                                                        now_time=now_time,
                                                        explicit=False) 
        return results
    
    def export_graph_data(self) -> Dict:
        """Export graph structure for visualization"""
        nodes = []
        edges = []
        
        for node_id, node in self.context_nodes.context_graph.items():
            if node is None:  
                continue
                
            is_root = node.sparent is None
            num_children = len(node.children)
            
            nodes.append({
                "id": node_id,
                "label": node.data.message[:50] + "..." if len(node.data.message) > 50 else node.data.message,
                "full_message": node.data.message,
                "role": node.data.role,
                "timestamp": node.data.timestamp.isoformat(),
                "is_root": is_root,
                "size": 10 + (num_children * 2),  # Bigger nodes for more children
                "color": "#ff6b6b" if is_root else "#4dabf7",  # Red for roots, blue for others
                "group": self._find_root_id(node)  # Group by conversation thread
            })
            

            if node.sparent:
                edges.append({
                    "source": node.sparent.id,
                    "target": node_id,
                    "weight": 1
                })


        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "root_nodes": sum(1 for n in nodes if n["is_root"]),
            "max_depth": self._calculate_max_depth(),
            "largest_thread": max(
                (len(self.get_conversation_story(n["id"])) for n in nodes if n["is_root"]),
                default=0
            )
        }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": stats,
            "export_time": datetime.now().isoformat()
        }

    def _find_root_id(self, node: HistoryNode) -> int:
        """Find the root node ID for grouping"""
        current = node
        while current.sparent is not None:
            current = current.sparent
        return current.id

    def _calculate_max_depth(self) -> int:
        """Calculate the maximum depth of any conversation thread"""
        max_depth = 0
        for node_id in self.root_nodes:
            if node_id in self.context_nodes.context_graph:
                depth = self._get_tree_depth(self.context_nodes.context_graph[node_id])
                max_depth = max(max_depth, depth)
        return max_depth

    def _get_tree_depth(self, node: HistoryNode, current_depth: int = 0) -> int:
        """Recursively calculate tree depth"""
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    

        

