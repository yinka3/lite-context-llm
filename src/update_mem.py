from typing import Dict, List, Optional, Set
import os
import numpy as np
import pytz
import numpy as np
from datetime import datetime
from _types import ContextGraph, EventData, HistoryNode, TimedConfigType
from strategy import DynamicSimilarity, TimeBased
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
    def __init__(self):
        self.history: List[HistoryNode] = []
        self._ephemeral_history: List[HistoryNode] = []
        self.context_nodes: ContextGraph = ContextGraph(context_graph={})
        self.root_nodes: Set[HistoryNode] = set()
        self._user_event_cnt = 0
        self.top_events: List[HistoryNode] = []

        self.vectorDB = ChromaClient()
        self.timebased_memory = TimeBased()
        self.dynamic_memory = DynamicSimilarity()
        self.storage = Storage(self)

        self.storage._load_from_disk()

    async def initialize(self):
        await self.vectorDB.initialize()
    
    async def add_to_history(self, event: EventData, context_graph: Optional[ContextGraph] = None):
        
        try:
            if event.role.lower() == "user":
                self._user_event_cnt += 1
            node_id = len(self.history)
            new_node = HistoryNode(node_id, event)
            self.context_nodes.context_graph[node_id] = new_node
            await self.vectorDB.add_event(new_node)
            self.history.append(new_node)

            if len(self.history) % 20 == 0:
                self.storage._save_to_disk()
                
            if context_graph is None:
                await self._update_context(new_node)
        except Exception as e:
            print("issue went wrong: {e}")
            return

    async def _update_context(self, event: HistoryNode):
        
        if event.data.role != "user":
            return
        
        try:
            results = await self.vectorDB.query(event.data.message)
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
                initial_similarity = 1 - results['distances'][0][i]
                

                time_score = await self.timebased_memory.apply(
                    initial_similarity=initial_similarity,
                    event=event,
                    parent_event=candidate_node
                )
                
                if time_score > best_score:
                    best_score = time_score
                    best_parent_id = candidate_id
            
            if best_parent_id is None:
                print(f"No suitable parent found for event {event.id}, creating root node")
                event.sparent = None
                return
            
            
            dynamic_threshold = None
            try:
                dynamic_threshold = await self.dynamic_memory.apply(
                    history=self.history,
                    vector_db=self.vectorDB
                )
            except:
                dynamic_threshold = 0.75
            
            if best_score >= dynamic_threshold:
                best_parent_event = self.context_nodes.context_graph[best_parent_id]
                event.sparent = best_parent_event
                best_parent_event.children.append(event)
            else:
                print(f"Similarity {best_score:.3f} below threshold {dynamic_threshold:.3f}, making root node")
                event.sparent = None
                
        except Exception as e:
            print(f"Critical error in context building for event {event.id}: {e}")
            self.root_nodes.add(event)

    async def get_relevant_context(self, query: str, max_results: int = 5) -> List[Dict]:
        """Get relevant historical context for a query"""
        try:
            results = await self.vectorDB.query(query, n_results=max_results)
            
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
    

        

