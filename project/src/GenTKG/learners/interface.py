from typing import Protocol
from typing import List

from project.src.GenTKG.models import TemporalLogicRule


class ILearner(Protocol):
    def learn(self):
        """
        Initialize the learning process. This method should definately be cached if possible.
        """
        pass

    def get_rules(self) -> List[TemporalLogicRule]:
        """
        Get the learned rules.
        """
        pass
