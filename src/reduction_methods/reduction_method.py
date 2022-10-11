from abc import ABC, abstractmethod
from src.database import Database


class ReductionMethod(ABC):
    """
    A class to represent Database reduction method
    """

    @abstractmethod
    def reduce(self, db: Database) -> Database:
        """
        Method for reducing the Database
        :param db: Database for reduction
        :return: Reduced database
        """
        pass
