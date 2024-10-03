from collections import defaultdict
from typing import List

import pandas as pd
from project.src.GenTKG.models import TemporalRandomWalk, Quadruple, TemporalLogicRule


class TemporalLogicRuleLearner:
    def __init__(self):
        self.temporal_rules = defaultdict(list)

    def _extract_temporal_random_walks(
        self, df_quadruples: pd.DataFrame  # List[Quadruple]
    ) -> List[TemporalRandomWalk]:
        print(
            f"Extracting temporal random walks from {len(df_quadruples)} quadruples..."
        )

        # Group by subject-object pairs and sort by timestamp in descending order
        grouped = df_quadruples.groupby(["subject_id", "object_id"])

        temporal_random_walks: List[TemporalRandomWalk] = []
        for (subject, obj), group in grouped:
            group_sorted = group.sort_values(by="timestamp", ascending=False)

            quadruples = [
                Quadruple(
                    subject_id=row["subject_id"],
                    relation_id=row["relation_id"],
                    object_id=row["object_id"],
                    timestamp=row["timestamp"],
                )
                for _, row in group_sorted.iterrows()
            ]

            for i in range(len(quadruples) - 1):
                # Ensure that the body timestamp is less than the head timestamp
                if quadruples[i].timestamp <= quadruples[i + 1].timestamp:
                    continue
                temporal_delta = quadruples[i].timestamp - quadruples[i + 1].timestamp
                temporal_random_walks.append(
                    TemporalRandomWalk(
                        relation_id_head=quadruples[i].relation_id,
                        relation_id_body=quadruples[i + 1].relation_id,
                        temporal_delta=temporal_delta,
                    )
                )

        return temporal_random_walks

    def _build_temporal_logic_rules(
        self, temporal_random_walks: List[TemporalRandomWalk]
    ) -> List[TemporalLogicRule]:
        RULE_BODY_COLUMNS = ["relation_id_body", "temporal_delta"]

        print(
            f"Building temporal logic rules from {len(temporal_random_walks)} random walks..."
        )
        # 1. Create a DataFrame from the temporal random walks
        df_temporal_random_walks = pd.DataFrame(
            [trw.model_dump() for trw in temporal_random_walks]
        )

        # 2. Calculate the support for the body and head relations.
        #       - rule_body is the number of occurrences of the body relation
        #       - rule_head is the number of occurrences of the head relation, given the body relation
        rule_body_support = df_temporal_random_walks.groupby(
            [*RULE_BODY_COLUMNS]
        ).size()
        rule_head_support = df_temporal_random_walks.groupby(
            ["relation_id_head", *RULE_BODY_COLUMNS]
        ).size()

        # 3. Compute the confidence of the rule
        confidence = rule_head_support.div(rule_body_support).rename("confidence")

        df_temporal_logic_rules = (
            pd.DataFrame(confidence)
            .sort_values("confidence", ascending=False)
            .reset_index()
        )
        return [
            TemporalLogicRule(
                relation_id_head=(row["relation_id_head"],),
                relation_id_body=(row["relation_id_body"],),
                temporal_delta=row["temporal_delta"],
                confidence=row["confidence"],
            )
            for _, row in df_temporal_logic_rules.iterrows()
        ]

    def run(self, df_quadruples: pd.DataFrame) -> List[TemporalLogicRule]:
        temporal_random_walks = self._extract_temporal_random_walks(df_quadruples)
        temporal_rules = self._build_temporal_logic_rules(temporal_random_walks)
        return temporal_rules
