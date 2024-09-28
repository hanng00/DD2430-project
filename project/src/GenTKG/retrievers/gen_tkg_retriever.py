from project.src.GenTKG.learners import TemporalLogicRuleLearner
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
        training_data: pd.DataFrame,
        retrieval_data: pd.DataFrame,
        cache_path: str | None = None,
    ) -> None:
        self.w = w  # Time window length, [h]
        self.N = N  # Max number of facts to retrieve
        self.training_data = training_data  # Columns as List[QuadrupleDecoded] which the temporal rules will be learned from
        self.retrieval_data = retrieval_data  # Columns as List[QuadrupleDecoded] which the facts will be retrieved from

        self.temporal_logic_rules: List[TemporalLogicRule] = (
            self._create_temporal_logic_rules(cache_path=cache_path)
        )
        self.df_temporal_rules = self._temporal_rules_to_df(self.temporal_logic_rules)

    def retrieve_facts(self, query: TKGQuery, verbose: bool = False) -> List[TKGFact]:
        # 1. Extract out the queryset. That is, within the time window, for the given subject
        time_window_start = query.timestamp - self.w
        time_window_end = query.timestamp
        df_queryset = self.retrieval_data.query(
            f"{time_window_start} <= timestamp and timestamp < {time_window_end} and subject_id == {query.subject_id}"
        )
        if verbose:
            print(f"Retrieved {len(df_queryset)} facts in the time window.")

        # 2. Retrieve the facts based on the rule head
        df_facts_rule_head = (
            df_queryset.query(f"relation_id == {query.relation_id}")
            .sort_values(by="timestamp", ascending=False)
            .head(self.N)
        )

        if verbose:
            print(f"Retrieved {len(df_facts_rule_head)} facts based on the rule head.")

        # Check if we have enough facts based on the rule head.
        if len(df_facts_rule_head) == self.N:
            return self._serialize_facts(df_facts_rule_head)

        remaining_N = self.N - len(df_facts_rule_head)

        # 3. Next, we apply the learned temporal logic rules to retrieve the rule body facts
        #    We only want to consider the rules that are relevant to the head relation,
        #    and where the time delta is within the time window.

        df_temporal_rules_queryset = self.df_temporal_rules.query(
            f"relation_id_head == {query.relation_id} and temporal_delta <= {self.w}"
        ).sort_values(by="confidence", ascending=False)

        # Get the most confident rule for each relation pair.
        # This is a simple way to ensure that we only consider the most confident rule for each relation pair,
        # which may have different time_deltas.
        df_temporal_rules_relevant = df_temporal_rules_queryset.loc[
            df_temporal_rules_queryset.groupby(
                ["relation_id_body", "relation_id_head"]
            )["confidence"].idxmax()
        ]

        if verbose:
            print(
                f"Found {len(df_temporal_rules_relevant)} relevant temporal rules to the head."
            )
        df_facts_rule_body = (
            df_temporal_rules_relevant.merge(
                df_queryset.query(
                    f"relation_id != {query.relation_id}"
                ),  # Exclude the head relation
                left_on="relation_id_body",
                right_on="relation_id",
                how="inner",
            )
            .sort_values(by="confidence", ascending=False)
            .head(remaining_N)
        )

        if verbose:
            print(f"Retrieved {len(df_facts_rule_body)} facts based on the rule body.")

        # 4. Combine the facts from the rule head and body
        df_facts_combined = pd.concat(
            [df_facts_rule_head, df_facts_rule_body]
        ).sort_values(by="timestamp", ascending=True)
        return self._serialize_facts(df_facts_combined)

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

    def _create_temporal_logic_rules(
        self, cache_path: str | None
    ) -> List[TemporalLogicRule]:
        try:
            if cache_path is not None:
                return self._read_from_cache(cache_path)
        except FileNotFoundError:
            pass

        # If not, learn the rules
        rule_learner = TemporalLogicRuleLearner()
        temporal_logic_rules = rule_learner.run(self.training_data)
        self._cache_temporal_rules(temporal_logic_rules, cache_path)
        return temporal_logic_rules

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

    def _read_from_cache(self, cache_path: str):
        df_temporal_rules = pd.read_json(cache_path)
        temporal_rules = [
            TemporalLogicRule(
                relation_id_head=row["relation_id_head"],
                relation_id_body=row["relation_id_body"],
                temporal_delta=row["temporal_delta"],
                confidence=row["confidence"],
            )
            for _, row in df_temporal_rules.iterrows()
        ]
        return temporal_rules
