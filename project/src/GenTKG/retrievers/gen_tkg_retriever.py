from project.src.GenTKG.learners.interface import ILearner
from project.src.GenTKG.retrievers.interface import IFactsRetriever
from typing import List

import pandas as pd
from project.src.GenTKG.models import TKGFact, TKGQuery, TemporalLogicRule


class GenTKGFactsRetriever(IFactsRetriever):
    """
    According to Algorithm 1 in the paper, we represent the temporal knowledge graph G
    as a serialized pandas DataFrame with the following columns:
        - subject_id: int
        - relation_id: int
        - object_id: int
        - timestamp: int

    It first creates the temporal logic rules using the TemporalLogicRuleLearner.
    Then, it retrieves the facts based on the query using the rules.

    """

    def __init__(
        self,
        w: int,
        N: int,
        retrieval_data: pd.DataFrame,
        learner: ILearner,
    ) -> None:
        self.w = w  # Time window length, [h]
        self.N = N  # Max number of facts to retrieve
        self.retrieval_data = retrieval_data  # Columns as List[QuadrupleDecoded] which the facts will be retrieved from
        self.learner = learner  # Learner to learn and retreive the temporal logic rules

        self.temporal_logic_rules = self.learner.get_rules()
        self.df_temporal_rules = self._temporal_rules_to_df(self.temporal_logic_rules)

    def retrieve_facts(self, query: TKGQuery, verbose: bool = False) -> List[TKGFact]:
        # Step 1: Extract facts within the time window
        df_queryset = self._get_queryset_within_time_window(query)
        if verbose:
            print(f"Retrieved {len(df_queryset)} facts in the time window.")

        # Step 2: Retrieve facts based on the rule head
        df_facts_rule_head = self._retrieve_rule_head_facts(df_queryset, query)
        if verbose:
            print(f"Retrieved {len(df_facts_rule_head)} facts based on the rule head.")

        # Check if we have enough facts based on the rule head
        if len(df_facts_rule_head) == self.N:
            return self._serialize_facts(df_facts_rule_head)

        remaining_N = self.N - len(df_facts_rule_head)

        # Step 3: Retrieve relevant temporal rules
        df_temporal_rules_relevant = self._get_relevant_temporal_rules(query)
        if verbose:
            print(
                f"Found {len(df_temporal_rules_relevant)} relevant temporal rules to the head."
            )

        # Step 4: Retrieve facts based on the rule body
        df_facts_rule_body = self._retrieve_rule_body_facts(
            df_queryset, df_temporal_rules_relevant, query, remaining_N
        )
        if verbose:
            print(f"Retrieved {len(df_facts_rule_body)} facts based on the rule body.")

        # Step 5: Combine the facts from rule head and body
        df_facts_combined = self._combine_facts(df_facts_rule_head, df_facts_rule_body)

        return self._serialize_facts(df_facts_combined)

    def _get_queryset_within_time_window(self, query: TKGQuery) -> pd.DataFrame:
        time_window_start = query.timestamp - self.w
        time_window_end = query.timestamp
        df_queryset = self.retrieval_data.query(
            f"{time_window_start} <= timestamp and timestamp < {time_window_end} and subject_id == {query.subject_id}"
        )
        return df_queryset

    def _retrieve_rule_head_facts(
        self, df_queryset: pd.DataFrame, query: TKGQuery
    ) -> pd.DataFrame:
        df_facts_rule_head = (
            df_queryset.query(f"relation_id == {query.relation_id}")
            .sort_values(by="timestamp", ascending=False)
            .head(self.N)
        )
        return df_facts_rule_head

    def _get_relevant_temporal_rules(self, query: TKGQuery) -> pd.DataFrame:
        df_temporal_rules_queryset = self.df_temporal_rules.query(
            f"relation_id_head == {query.relation_id} and temporal_delta <= {self.w}"
        ).sort_values(by="confidence", ascending=False)

        df_temporal_rules_relevant = df_temporal_rules_queryset.loc[
            df_temporal_rules_queryset.groupby(
                ["relation_id_body", "relation_id_head"]
            )["confidence"].idxmax()
        ]
        return df_temporal_rules_relevant

    def _retrieve_rule_body_facts(
        self,
        df_queryset: pd.DataFrame,
        df_temporal_rules_relevant: pd.DataFrame,
        query: TKGQuery,
        remaining_N: int,
    ) -> pd.DataFrame:
        df_facts_rule_body = (
            df_temporal_rules_relevant.merge(
                df_queryset.query(f"relation_id != {query.relation_id}"),
                left_on="relation_id_body",
                right_on="relation_id",
                how="inner",
            )
            .sort_values(by="confidence", ascending=False)
            .head(remaining_N)
        )
        return df_facts_rule_body

    def _combine_facts(
        self, df_facts_rule_head: pd.DataFrame, df_facts_rule_body: pd.DataFrame
    ) -> pd.DataFrame:
        df_facts_combined = pd.concat(
            [df_facts_rule_head, df_facts_rule_body]
        ).sort_values(by="timestamp", ascending=True)
        return df_facts_combined

    def _temporal_rules_to_df(
        self, temporal_rules: List[TemporalLogicRule]
    ) -> pd.DataFrame:
        df_temporal_rules = pd.DataFrame([rule.model_dump() for rule in temporal_rules])
        df_temporal_rules["relation_id_head"] = df_temporal_rules[
            "relation_id_head"
        ].apply(lambda x: x[0])
        df_temporal_rules["relation_id_body"] = df_temporal_rules[
            "relation_id_body"
        ].apply(lambda x: x[0])
        return df_temporal_rules

    def _serialize_facts(self, df: pd.DataFrame) -> List[TKGFact]:
        return [
            TKGFact(
                timestamp=row["timestamp"],
                subject=row["subject"],
                relation=row["relation"],
                object=row["object"],
                object_id=row["object_id"],
            )
            for _, row in df.iterrows()
        ]

    def _cache_temporal_rules(
        self, temporal_rules: List[TemporalLogicRule], cache_path: str
    ):
        df_temporal_rules = pd.DataFrame([tr.model_dump() for tr in temporal_rules])
        df_temporal_rules.to_json(cache_path, index=False)
        print(f"Temporal rules cached to {cache_path}")
