import heapq
from typing import Dict, List, Optional, Set
import os

import numpy as np
import pytz
import numpy as np
from datetime import datetime
from _types import ContextGraph, EventData, HistoryNode, TimedConfigType
from strategy import DynamicSimilarity, TimeBased
from vectorDB import ChromaClient
import spacy

try:
    mspacy_medium = spacy.load('en_core_web_md')
    mspacy_large = spacy.load('en_core_web_lg')
except Exception as e:
    print(f"Spacy model not load")


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
    
    async def initialize(self):
        await self.vectorDB.initialize()
    
    async def add_to_history(self, event: EventData, context_graph: Optional[ContextGraph]):
        
        try:
            if event.role.lower() == "user":
                self._user_event_cnt += 1
            node_id = len(self.history) - 1
            new_node = HistoryNode(node_id, event)
            self.context_nodes.context_graph[node_id] = new_node
            await self.vectorDB.add_event(new_node)
            self.history.append(new_node)
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
            initial_similarity = 0
            
            for i, node_id in enumerate(results['ids'][0]):
                candidate_id = int(node_id)

                if candidate_id == event.id:
                    continue
                    
                if candidate_id not in self.context_nodes.context_graph:
                    continue
                    
                if self._would_create_cycle(event, candidate_id):
                    continue
                
                best_parent_id = candidate_id
                initial_similarity = 1 - results['distances'][0][i]
                break
            
            if best_parent_id is None:
                print(f"No suitable parent found for event {event.id}, creating root node")
                self._make_root_node(event)
                return
            
            try:
                best_parent_event = self.context_nodes.context_graph[best_parent_id]
                final_score = await self.timebased_memory.apply(
                    initial_similarity=initial_similarity,
                    event=event,
                    parent_event=best_parent_event
                )
                
                dynamic_threshold = await self.dynamic_memory.apply(
                    history=self.history,
                    vector_db=self.vectorDB
                )
                
            except Exception as e:
                print(f"Error calculating similarity scores: {e}")
                final_score = initial_similarity
                dynamic_threshold = 0.75
            
            if final_score >= dynamic_threshold:
                event.sparent = best_parent_event
                best_parent_event.children.append(event)
            else:
                print(f"Similarity {final_score:.3f} below threshold {dynamic_threshold:.3f}, making root node")
                event.sparent = None
                
        except Exception as e:
            print(f"Critical error in context building for event {event.id}: {e}")
            self.root_nodes.add(event.id)


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
    

    # this is an experiment, not sure if this will actually work or how to even use it propery for my intention
    # async def _ephemeral_context(self, rant_event: EventData):
    #     forked_graph = copy.deepcopy(self.context_nodes)
    #     copy_list = copy.deepcopy(self.top_events)
    #     _id = -1
    #     new_node = HistoryNode(id=_id, data=rant_event)

    #     best_top_event: List[HistoryNode] = []
    #     for top_eve in copy_list:
    #         embeddings = await self.vector_db.get_event(ids=[top_eve.id, rant_event])
    #         similarity_matrix = cosine_similarity(embeddings)
    #         if similarity_matrix > 0.7:
    #             best_top_event.append(top_eve)
        
    #     for eve in best_top_event:
    #         eve.children.append(new_node)
        

    #     # Do stuff with it

    #     # empty out the epharamel list
    #     self._ephemeral_history.clear()

    # def dfs_search(self, start_node: HistoryNode):

    #     visited = set()

    #     def dfs(current_node_id: int):
    #         visited.add(current_node_id)
    #         current_node = self.context_nodes.context_graph.get(current_node_id)
    #         if not current_node:
    #             return
            
    #         for child_node in current_node.children:
    #             if child_node.id not in visited:
    #                 dfs(child_node.id)

    #     dfs(start_node.id)
    #     return len(visited) - 1
    

        

