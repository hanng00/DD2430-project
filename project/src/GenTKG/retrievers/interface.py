from typing import List, Protocol

from project.src.GenTKG.models import TKGFact, TKGQuery


class IFactsRetriever(Protocol):
    def retrieve_facts(self, query: TKGQuery, verbose: bool) -> List[TKGFact]:
        pass
