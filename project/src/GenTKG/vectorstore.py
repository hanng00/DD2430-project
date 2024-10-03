from typing import Protocol
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import pandas as pd


class IVectorstore(Protocol):
    def insert(self, relations: List[str]) -> None:
        """
        Inserts relations into the vectorstore.
        """
        pass

    def get_topk_similar(self, relation: str, k: int) -> List[str]:
        """
        Retrieves the top-k most similar relations to the given relation
        wrt to cosine similarity.
        """
        pass

    def update_cache(self) -> None:
        """
        Updates the cache file with the current state of the vectorstore.
        """
        pass


class SimpleVectorstore:
    def __init__(self, cache_path: str, openai_api_key: str):
        self.openai = OpenAI(api_key=openai_api_key)
        self.cache_path = cache_path

        self.store = self._get_cached_store()

    def insert(self, relations: List[str]) -> None:
        # First, ensure we are not inserting duplicates
        new_relations = [
            relation
            for relation in relations
            if relation not in self.store["relation"].tolist()
        ]
        if len(new_relations) == 0:
            print("No new relations to insert")
            return
        print(f"Inserting {len(new_relations)} new relations")

        embeddings = self._get_embeddings(new_relations)
        for relation, embedding in zip(relations, embeddings):
            self._insert_vector(relation, embedding)

    def _get_embeddings(self, relations: List[str]) -> List[float]:
        response = self.openai.embeddings.create(
            dimensions=512,
            model="text-embedding-3-large",
            input=relations,
        )
        embeddings_object = response.data
        embedding_vectors = [obj.embedding for obj in embeddings_object]

        return embedding_vectors

    def _insert_vector(self, relation: str, vector: List[float]) -> None:
        """
        Inserts a relation and its corresponding vector into the vectorstore.
        """
        new_entry = pd.DataFrame([{"relation": relation, "vector": vector}])
        self.store = pd.concat([self.store, new_entry], ignore_index=True)

    def get_topk_similar(self, relation: str, k: int) -> List[str]:
        """
        Retrieves the top-k most similar relations to the given relation
        with respect to cosine similarity.
        """
        # Get the vector for the given relation
        target_embedding = self._get_embeddings([relation])[0]

        # Calculate cosine similarity between the target vector and all vectors in the store
        vectors = self.store["vector"].tolist()
        similarities = cosine_similarity([target_embedding], vectors)[0]

        # Sort based on similarity scores and retrieve the top-k indices
        top_k_indices = similarities.argsort()[-k:][::-1]

        # Get the top-k similar relations based on indices
        top_k_relations = self.store.iloc[top_k_indices]["relation"].tolist()

        return top_k_relations

    def _get_cached_store(self) -> pd.DataFrame:
        # Try to read from cache. The data file should be a CSV file.
        try:
            store = pd.read_json(self.cache_path)
        except FileNotFoundError:
            store = pd.DataFrame(columns=["relation", "vector"])

        return store

    def update_cache(self) -> None:
        self.store.to_json(self.cache_path, orient="records")
