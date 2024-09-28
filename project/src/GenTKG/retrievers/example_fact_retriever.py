import re
from typing import List
from project.src.GenTKG.models import TKGFact, TKGQuery
from project.src.GenTKG.retrievers.interface import IFactsRetriever


class ExampleFactsRetriever(IFactsRetriever):
    def __init__(self) -> None:
        super().__init__()

        # The extracted task input from the original paper
        self.example_facts = """
        93: [Abdulrahman, Make_statement, 8092.Government_(Nigeria)]
        113: [Abdulrahman, Make_statement, 8092.Government_(Nigeria)]
        162: [Abdulrahman, Praise_or_endorse, 15546.Muslim_(Nigeria)]
        197: [Abdulrahman, Consult, 8488.Governor_(Nigeria)]
        197: [Abdulrahman, Make_statement, 8092.Government_(Nigeria)]
        228: [Abdulrahman, Praise_or_endorse, 15414.Muhammadu_Buhari]
        270: [Abdulrahman, Make_an_appeal_or_request, 3835.Citizen_(Nigeria)]
        """

    def retrieve_facts(self, query: TKGQuery) -> List[TKGFact]:
        """Retrieve the example rules"""
        return self._parse_retreived_facts(self.example_facts)

    def _parse_retreived_facts(self, task_input_text: str) -> List[TKGFact]:
        """
        Convert the task input text into a list of TKG rules and a TKG query.
        Each dictionary corresponds to a quadruplet with keys: 'time', 'subject', 'relation', 'object_label'.
        """
        # Split the task input text into individual lines
        lines = task_input_text.strip().splitlines()

        task_input_list = []

        # Regular expression to extract the data in the format "{time}: [{subject}, {relation}, {object_label}.{object}]"
        pattern = r"(\d+): \[(.*?), (.*?), (\d+)\.(.*?)\]"

        for line in lines:
            match = re.match(pattern, line.strip())

            if not match:
                continue

            time, subject, relation, object_label, obj = match.groups()
            data_object = TKGFact(
                timestamp=time,
                subject=subject,
                relation=relation,
                object=obj,
                object_id=int(object_label),
            )

            task_input_list.append(data_object)

        return task_input_list
