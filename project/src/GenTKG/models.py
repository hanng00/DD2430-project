from pydantic import BaseModel, Field
from typing import Dict, List, Tuple
from typing import Protocol
from datetime import datetime, timedelta

import humanize


def hours_to_text(start_h: int, end_h: int):
    return humanize.naturaltime(datetime.now() - timedelta(hours=(end_h - start_h)))


class TKGFact(BaseModel):
    timestamp: int
    subject: str
    relation: str
    object: str
    object_id: int

    def __str__(self):
        return f"{self.timestamp}:[{self.subject}, {self.relation}, {self.object_id}:{self.object}]"

    def to_relative_string(self, reference_timestamp: int):
        humanized_time_ago = hours_to_text(self.timestamp, reference_timestamp)
        return f"[{humanized_time_ago}]:[{self.subject}, {self.relation}, {self.object_id}:{self.object}]"


class TKGQuery(BaseModel):
    timestamp: int
    subject_id: int
    subject: str
    relation_id: int
    relation: str

    def __str__(self):
        return f"{self.timestamp}:[{self.subject}, {self.relation},"

    def to_relative_string(self, reference_timestamp: int):
        humanized_time_ago = hours_to_text(self.timestamp, reference_timestamp)
        return f"[{humanized_time_ago}]:[{self.subject}, {self.relation},"


class Quadruple(BaseModel):
    subject_id: int = Field(..., description="ID of the subject entity")
    relation_id: int = Field(..., description="ID of the relation type")
    object_id: int = Field(..., description="ID of the object entity")
    timestamp: int = Field(..., description="Timestamp of the interaction")


class TemporalRandomWalk(BaseModel):
    relation_id_head: int = Field(..., description="ID of the head relation")
    relation_id_body: int = Field(..., description="ID of the body relation")
    temporal_delta: int = Field(
        ..., description="Temporal difference between the head and body relations"
    )


class TemporalLogicRule(BaseModel):
    relation_id_head: Tuple[int] = Field(
        ..., description="Tuple representing the head of the rule (relation_id)"
    )
    relation_id_body: Tuple[int] = Field(
        ..., description="Tuple representing the body of the rule (relation_id)"
    )
    temporal_delta: int = Field(
        ..., description="Temporal difference between the head and body relations"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the rule"
    )
