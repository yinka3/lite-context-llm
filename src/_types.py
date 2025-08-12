from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union



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
class TimedConfigType:
    name: str
    type: HistoryNode | Union[list, datetime]


