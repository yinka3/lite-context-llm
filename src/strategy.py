from abc import ABC, abstractmethod
import numpy as np
from _types import HistoryNode
from vectorDB import ChromaClient
from sklearn.metrics.pairwise import cosine_similarity

class LinkingStrategy(ABC):
    @abstractmethod
    async def apply(self, **kwargs):
        raise NotImplementedError


class TimeBased(LinkingStrategy):
    def __init__(self, decay_type="exponential", tune=0.5, client: ChromaClient = None):
        self.client = client
        self.tune = tune
        self.decay_type = decay_type
    
    async def apply(self, *, initial_similarity: float, event: HistoryNode, parent_event: HistoryNode) -> float:
        time_difference = (event.data.timestamp - parent_event.data.timestamp).total_seconds()
        

        hours_diff = time_difference / 3600
        
        if self.decay_type == "exponential":
            decay_factor = np.exp(-self.tune * hours_diff)
        elif self.decay_type == "power":
            decay_factor = 1 / (1 + hours_diff) ** self.tune
        elif self.decay_type == "sigmoid":
            decay_factor = 1 / (1 + np.exp(self.tune * (hours_diff - 12)))
        elif self.decay_type == "stepped":
            if hours_diff < 1:
                decay_factor = 1.0
            elif hours_diff < 24:
                decay_factor = 0.8
            elif hours_diff < 168: 
                decay_factor = 0.5
            else:
                decay_factor = 0.2
        
        decay_factor = max(decay_factor, 0.1)        
        final_score = initial_similarity * decay_factor
        return final_score
            
            
            
class DynamicSimilarity(LinkingStrategy):
    def __init__(self, base_threshold=0.75, tune=0.1):
        self.tune = tune
        self.base_threshold = base_threshold
        self.min_window = 3
        self.max_window = 15
        self.growth_factor = 2.5

    def _calculate_window_size(self, history_length: int) -> int:
        if history_length <= self.min_window:
            return self.min_window

        dynamic_size = self.min_window + self.growth_factor * np.log(history_length)
        return int(np.clip(dynamic_size, self.min_window, self.max_window))

    async def apply(self, *, history: list, vector_db: ChromaClient) -> float:
        window_size = self._calculate_window_size(len(history))
        
        if len(history) < window_size:
            return self.base_threshold

       
        recent_messages_ids = [
             str(event.id) for event in history[-window_size:] if event.data.role == "user"
        ]
        
        embeddings = await vector_db.get_event(ids=recent_messages_ids)

        if not embeddings or len(embeddings) < 2:
            return self.base_threshold
        similarity_matrix = cosine_similarity(embeddings)
        
        iu = np.triu_indices(len(similarity_matrix), k=1)
        avg_similarity = np.mean(similarity_matrix[iu])

        adjusted_threshold = self.base_threshold - (avg_similarity * self.tune)
        return np.clip(adjusted_threshold, 0.65, 0.85)