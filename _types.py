from dataclasses import dataclass
from datetime import datetime
from typing import List, Union

from update_mem import HistoryNode

@dataclass
class EventData:
    role: str
    message: str

@dataclass
class ListHistory:
    histories: List[HistoryNode]

@dataclass
class TimedConfigType:
    name: str
    type: HistoryNode | Union[list, datetime]


