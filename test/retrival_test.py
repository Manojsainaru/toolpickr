import os
import sys
from dotenv import load_dotenv
from tools_definitions import tools
from queries import queries
from typing import List, Optional
from toolpickr.core.tool import ToolDefinition
from toolpickr.core.registry import ToolRegistry
from toolpickr.embeddings.gemini import GeminiEmbeddings
from toolpickr.embeddings.renderer import render_tool_text
from toolpickr.vectorstores.faiss import FaissVectorStore
from toolpickr.retrieval.flat import FlatRetriever
from toolpickr.vectorstores.base import SearchResult

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolpickr.pickr import ToolPickr
from toolpickr.core.tool import ToolDefinition

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

print("Initializing ToolPickr...")
registry = ToolRegistry() # think not needed
embeddings = GeminiEmbeddings(api_key=os.getenv("GEMINI_API_KEY"))
vector_store = FaissVectorStore(dimension=1536)
retriever = FlatRetriever(embeddings, vector_store)
is_built = False

for t in tools:
    # think not needed
    registry.register_tool(t)

# for test
def build_query_index():
    print("Building query index...")
    query_vectors = embeddings.embed_batch(queries)
    #query_vector_store.add_vectors(vectors=query_vectors, tool_names=queries)
    vector_store.save_query_vectors_to_disk(query_vectors, "faiss_index", "queries")
    print("Query index built!")

def build_embedding_index():
    print("Building index...")

    # Build Embedding Index
    names = []
    texts = []
            
    # Format each tool into a recognizable string
    for tool in tools:
        names.append(tool.name)
        texts.append(render_tool_text(tool))
                
    # Get vectors from Gemini (batching is faster)
    print(f"Embedding {len(tools)} tools via Gemini...")
    vectors = embeddings.embed_batch(texts)
            
    # Save them in FAISS
    vector_store.add_vectors(vectors=vectors, tool_names=names)
    is_built = True
    print("ToolPickr build complete!")
    vector_store.save_local("faiss_index")
    print("Index Saved")
    
def main():
    #build_embedding_index()
    #build_query_index()
    
    # Load vectors and index
    query_vectors = vector_store.load_query_vectors_from_disk("faiss_index", "queries")
    vector_store.load_local("faiss_index")
    
    test_queries = queries
    
    # print("\n--- Testing Batch Search ---")
    # batch_results = vector_store.search_batch(query_vectors, k=2)
    # for query, results in zip(test_queries, batch_results):
    #     print(f"Batch Query: '{query}'")
    #     for res in results:
    #         print(f"  -> {res}")
            
    print("\n--- Testing Single Search ---")
    for query, query_vector in zip(test_queries, query_vectors):
        print(f"Single Query: '{query}'")
        # Retrieve the best matches
        single_results = vector_store.search(query_vector, k=5)
        for res in single_results:
            print(f"  -> {res}")

if __name__ == "__main__":
    main()
