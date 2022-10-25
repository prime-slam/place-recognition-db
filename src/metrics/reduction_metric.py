from abc import ABC, abstractmethod
from numbers import Number
from src.core import Database


class ReductionMetric(ABC):
    """
    A class to represent Database reduction method
    """

    @abstractmethod
    def evaluate(self, original_db: Database, filtered_db: Database) -> Number:
        """
        Method for calculating the result of the metric
        :param original_db: Original database
        :param filtered_db: Reduced database
        :return: Result of the metric
        """
        pass

    @abstractmethod
    def evaluate_many(
        self, original_db: Database, filtered_dbs: list[Database]
    ) -> list[Number]:
        """
        Method for calculating the results of the metric for list of filtered DBs
        :param original_db: Original database
        :param filtered_dbs: Reduced databases
        :return: Results of the metric
        """
        pass
