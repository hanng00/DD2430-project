from typing import List

import pandas as pd
from project.src.GenTKG.models import TKGFact, TKGQuery
from project.src.GenTKG.retrievers.interface import IFactsRetriever


class NaiveFactsRetriever(IFactsRetriever):
    """
    Naive Facts Retriever

    This class retrieves the facts based on the query.
    It retrieves the last N facts based on the subject and relation.
    """

    def __init__(
        self,
        training_data: pd.DataFrame,  # Columns as List[QuadrupleDecoded]
        w: int = 72,
        N: int = 50,
    ):
        self.training_data = training_data

        self.w = w  # Time window length, [h]
        self.N = N  # Number of facts to retrieve at most

    def retrieve_facts(self, query: TKGQuery) -> List[TKGFact]:
        """Retrieve the facts based on the query"""
        # Query the training data for the given subject and relation
        subject_id = query.subject_id
        timestamp = query.timestamp

        df_N_recent_events = (
            self.training_data.query(
                f"subject_id == {subject_id} and timestamp < {timestamp}"
            )
            .sort_values(by="timestamp", ascending=False)
            .head(self.N)  # Grab the most recent N events
            .sort_values(by="timestamp", ascending=True)  # Reverse, most recent last.
        )

        # Encode to TKGFact
        facts = [
            TKGFact(
                timestamp=row["timestamp"],
                subject=row["subject"],
                relation=row["relation"],
                object=row["object"],
                object_id=row["object_id"],
            )
            for _, row in df_N_recent_events.iterrows()
        ]

        return facts
