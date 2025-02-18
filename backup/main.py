import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained model for embedding generation
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample text data
documents = [
    "AI is transforming the world.",
    "Machine learning is a subset of AI.",
    "Vector databases help store embeddings.",
    "FAISS is great for similarity search."
]

# Convert text into vector embeddings
embeddings = model.encode(documents)

# Convert to NumPy array (FAISS requires NumPy format)
embeddings = np.array(embeddings).astype('float32')
print(embeddings)

# Initialize a FAISS index for similarity search
dimension = embeddings.shape[1]  # Size of the embedding vectors
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index

# Add vectors to the FAISS index
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")

query = "What is Vector databases?"
query_embedding = model.encode([query]).astype('float32')

# Search for the top 2 most similar documents
k = 2  # Number of results to retrieve
distances, indices = index.search(query_embedding, k)

# Print the most relevant documents
print("Query:", query)
print("\nTop matching results:")
for i in range(k):
    print(f"{i+1}. {documents[indices[0][i]]} (Distance: {distances[0][i]:.4f})")


