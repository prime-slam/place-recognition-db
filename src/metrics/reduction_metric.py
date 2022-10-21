from abc import ABC, abstractmethod
from src.core import Database


class ReductionMetric(ABC):
    @abstractmethod
    def evaluate(self, original_db: Database, filtered_db: Database) -> float:
        pass
