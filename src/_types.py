from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, TypedDict, Union



@dataclass
class EventData:
    role: str
    message: str
    timestamp: datetime

@dataclass
class HistoryNode:
    id: int
    data: EventData
    children: List['HistoryNode'] = field(default_factory=list)
    sparent: Optional['HistoryNode'] = None


@dataclass
class ContextGraph:
    context_graph: Dict[int, HistoryNode]

@dataclass
class Context:
    matched_message: str = ""
    role: str = ""
    timestamp: datetime
    similarity: float
    conversation_thread: List[Dict[str, str]]

@dataclass
class TimedConfigType:
    name: str
    type: HistoryNode | Union[list, datetime]


@dataclass
class ChromaQueryResult:
    ids: List[List[str]] = field(default_factory=list)
    documents: List[List[str]] = field(default_factory=list)
    metadatas: List[List[Dict]] = field(default_factory=list)
    distances: List[List[float]] = field(default_factory=list)

@dataclass
class TopicExtractionResult:
    intent: str = "unknown"
    main_topic: Optional[str] = None
    entities: Dict[str, str] = field(default_factory=dict)
    state_info: Dict[str, Dict[str, str]] = field(default_factory=dict)

@dataclass
class RelationReport:
    is_related: bool = False
    confidence_score: float = 0.0
    common_elements: Set[str] = field(default_factory=set)
    found_connections: List[str] = field(default_factory=list)

@dataclass
class SentimentResult:
    score: float = 0.0
    label: str = ""
    positive_count: float = 0.0
    negative_count: float = 0.0