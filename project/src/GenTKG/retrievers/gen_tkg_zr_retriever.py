from project.src.GenTKG.learners.interface import ILearner
from project.src.GenTKG.retrievers.interface import IFactsRetriever
from typing import List

import pandas as pd
from project.src.GenTKG.models import TKGFact, TKGQuery, TemporalLogicRule
from project.src.GenTKG.vectorstore import IVectorstore


# TODO - Very similar to the GenTKGFactsRetriever, consider refactoring.
class GenTKGZeroShotFactsRetriever(IFactsRetriever):
    """
    Extended to handle zero-shot learning.

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
        vectorstore: IVectorstore | None = None,
    ) -> None:
        self.w = w  # Time window length, [h]
        self.N = N  # Max number of facts to retrieve
        self.retrieval_data = retrieval_data  # Columns as List[QuadrupleDecoded] which the facts will be retrieved from
        self.learner = learner  # Learner to learn and retreive the temporal logic rules

        self.temporal_logic_rules = self.learner.get_rules()
        self.df_temporal_rules = self._temporal_rules_to_df(self.temporal_logic_rules)
        self.vectorstore = vectorstore

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

        # Step 3: Retrieve relevant temporal rules based on the query
        df_temporal_rules = self._get_relevant_temporal_rules(
            query_relation_id=query.relation_id
        )
        if verbose:
            print(
                f"Found {len(df_temporal_rules)} relevant temporal rules to the query."
            )

        # Step 4: Check if there are no relevant temporal rules
        # If so, we use the heuristics to map the query to rules we may seem fit.
        if len(df_temporal_rules) == 0:
            df_topk_temporal_rules = self._retrieve_topk_similar_temporal_rules(query)

            if verbose:
                print(
                    f"Found {len(df_topk_temporal_rules)} top-k similar temporal rules."
                )

            df_temporal_rules = pd.concat(
                [df_temporal_rules, df_topk_temporal_rules], axis=0
            )

        # Step 4: Retrieve facts based on the rule body
        # Facts retrieved from the rule head is omitted
        df_queryset = df_queryset.query(f"relation_id != {query.relation_id}")
        df_facts_rule_body = self._retrieve_facts_from_temporal_rules(
            df_queryset, df_temporal_rules
        ).head(remaining_N)
        if verbose:
            print(f"Retrieved {len(df_facts_rule_body)} facts based on the rule body.")

        # Step 5: Combine the facts from rule head and body
        df_facts_combined = self._combine_facts(df_facts_rule_head, df_facts_rule_body)

        return self._serialize_facts(df_facts_combined)

    def _retrieve_topk_similar_temporal_rules(
        self, query: TKGQuery, k: int = 3
    ) -> pd.DataFrame:
        query_relation = query.relation

        # 1. Get the top-k most similar relations to the query relation
        topk_similar_relations = self.vectorstore.get_topk_similar(query_relation, k)
        assert (
            len(topk_similar_relations) == k
        ), f"Expected {k} similar relations, but got {len(topk_similar_relations)}"
        # print(f"Top-k similar relations to {query_relation}: {topk_similar_relations}")

        # Remove the query relation from the list, if it exists
        topk_similar_relations = [
            relation
            for relation in topk_similar_relations
            if relation != query_relation
        ]

        # 2. Retrieve the equivalent relation ids
        relation2id = self.retrieval_data[["relation", "relation_id"]].drop_duplicates()
        topk_similar_relations_ids = relation2id.query(
            f"relation in @topk_similar_relations"
        )["relation_id"].tolist()

        # 3. Retrieve the temporal rules based on the similar relations
        dfs_temporal_rules = [
            self._get_relevant_temporal_rules(relation_id)
            for relation_id in topk_similar_relations_ids
        ]

        df_temporal_rules_combined = pd.concat(dfs_temporal_rules, axis=0)
        return df_temporal_rules_combined

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

    def _get_relevant_temporal_rules(
        self, query_relation_id: int  # TKGQuery["relation_id"]
    ) -> pd.DataFrame:
        # Retrieve temporal rules that are relevant to the query
        df_temporal_rules_queryset = self.df_temporal_rules.query(
            f"relation_id_head == {query_relation_id} and temporal_delta <= {self.w}"
        ).sort_values(by="confidence", ascending=False)

        # Get the most confident rule for each relation pair, per temporal_delta
        df_temporal_rules_relevant = df_temporal_rules_queryset.loc[
            df_temporal_rules_queryset.groupby(
                ["relation_id_body", "relation_id_head"]
            )["confidence"].idxmax()
        ]
        return df_temporal_rules_relevant

    def _retrieve_facts_from_temporal_rules(
        self,
        df_queryset: pd.DataFrame,
        df_temporal_rules: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        This takes a queryset and temporal rules and returns the facts based on the rules sorted by confidence.
        """
        df_facts_rule_body = df_temporal_rules.merge(
            df_queryset,
            left_on="relation_id_body",
            right_on="relation_id",
            how="inner",
        ).sort_values(by="confidence", ascending=False)
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
