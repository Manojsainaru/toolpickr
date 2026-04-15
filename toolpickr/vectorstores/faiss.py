# FAISSVectorStore
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional
from toolpickr.vectorstores.base import VectorStore, SearchResult

class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int, debug: bool = False):
        """
        Initializes an empty FAISS index in memory.
        IndexFlatIP uses Inner Product (ideal for cosine similarity if vectors are normalized).
        """
        self.dimension = dimension
        self.debug = debug
        self.index = faiss.IndexFlatIP(dimension)
        # Maps FAISS integer IDs (row numbers) to tool names
        self.id_to_name: Dict[int, str] = {}
        # Keeps track of the next available integer ID
        self._next_id = 0

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[FaissVectorStore - Debug] {msg}")

    def add_vectors(self, vectors: List[List[float]], tool_names: List[str], metadata: List[dict] = None) -> None:
        """Adds vectors to the FAISS index."""
        if not vectors:
            self._log("No vectors to add.")
            return

        self._log(f"Adding {len(vectors)} vectors to FAISS index.")

        if len(vectors) != len(tool_names):
            raise ValueError("Number of vectors must match number of tool names.")

        # Convert to numpy array of float32 (FAISS requirement)
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # If incomming dimention does not match the index dimention, 
        # and the index is empty, then recreate the index with the correct dimention
        # TODO: remove the manual declaration of dimentions
        d = vectors_np.shape[1]
        if self.dimension != d:
            if self.index.ntotal == 0:
                self.dimension = d
                self.index = faiss.IndexFlatIP(d)
            else:
                raise ValueError(f"Vector dimension {d} does not match the index dimension {self.dimension}.")

        # Add to FAISS. The row index determines its ID.
        self.index.add(vectors_np)

        # Keep track of which tool name corresponds to which row
        for name in tool_names:
            self.id_to_name[self._next_id] = name
            self._next_id += 1

    def search(self, query_vector: List[float], k: int = 5) -> List[SearchResult]:
        """Performs a similarity search."""
        if self.index.ntotal == 0:
            self._log("Search called but index is empty.")
            return []

        self._log(f"Searching index for 1 query vector (k={k})")
        # Convert query to numpy array, shaped as (1, dimension)
        query_np = np.array([query_vector], dtype=np.float32)

        # Search the index
        scores, inner_ids = self.index.search(query_np, k)

        results = []
        # scores and inner_ids are 2D arrays, fetch the first row
        for score, inner_id in zip(scores[0], inner_ids[0]):
            if inner_id != -1: # FAISS returns -1 if there aren't enough vectors
                tool_name = self.id_to_name[inner_id]
                results.append(SearchResult(tool_name=tool_name, score=float(score)))

        return results

    # Created for testing purposes
    def search_batch(self, query_vectors: List[List[float]], k: int = 5) -> List[List[SearchResult]]:
        """Performs a similarity search for multiple query vectors."""
        if self.index.ntotal == 0:
            return [[] for _ in query_vectors]

        # Convert query vectors to numpy array, shaped as (num_queries, dimension)
        query_np = np.array(query_vectors, dtype=np.float32)

        # Search the index
        scores, inner_ids = self.index.search(query_np, k)

        results = []
        # Iterate through each query's results
        for i in range(len(query_vectors)):
            query_results = []
            for score, inner_id in zip(scores[i], inner_ids[i]):
                if inner_id != -1: # FAISS returns -1 if there aren't enough vectors
                    tool_name = self.id_to_name[inner_id]
                    query_results.append(SearchResult(tool_name=tool_name, score=float(score)))
            results.append(query_results)

        return results


    # --- Disk Persistence Methods ---

    # For testing purposes
    def save_query_vectors_to_disk(self, query_vectors: List[List[float]], directory: str, name: str = "queries"):
        """Saves the query vectors to disk."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{name}.json"), "w") as f:
            json.dump(query_vectors, f)

    # For testing purposes
    def load_query_vectors_from_disk(self, directory: str, name: str = "queries") -> List[List[float]]:
        """Loads the query vectors from disk."""
        with open(os.path.join(directory, f"{name}.json"), "r") as f:
            return json.load(f)
    
    def save_local(self, directory: str, index_name: str = "tools"):
        """Saves the FAISS index and the id mapping to disk."""
        os.makedirs(directory, exist_ok=True)
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, f"{index_name}.faiss"))
        # Save the dictionary mapping (using standard json)
        import json
        with open(os.path.join(directory, f"{index_name}_mapping.json"), "w") as f:
            json.dump(self.id_to_name, f)

    def load_local(self, directory: str, index_name: str = "tools"):
        """Loads a previously saved FAISS index and id mapping from disk."""
        self.index = faiss.read_index(os.path.join(directory, f"{index_name}.faiss"))
        import json
        with open(os.path.join(directory, f"{index_name}_mapping.json"), "r") as f:
            # json saves keys as strings, we need them back as integers
            mapping = json.load(f)
            self.id_to_name = {int(k): v for k, v in mapping.items()}
            self._next_id = max(self.id_to_name.keys()) + 1 if self.id_to_name else 0
