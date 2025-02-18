import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg instead of GTK4

import faiss
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define one question
question = "How many players are in a cricket team?"

# Define multiple options (documents)
documents = [
    "There are 11 players in a cricket team.",  # Correct answer (to be determined automatically)
    "Basketball is played with 5 players per team.",
    "A kabaddi team consists of 7 players.",
    "A football team has 11 players.",
    "Hockey teams have 6 players on the ice."
]

# Combine question and options
all_texts = [question] + documents  # Question is the first item

# Convert text into vector embeddings
embeddings = model.encode(all_texts)
embeddings = np.array(embeddings).astype('float32')

# Reduce dimensions from 384D to 3D using PCA
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# Initialize FAISS index and add only document vectors
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings[1:])  # Add only options (documents) to the FAISS index

# Perform similarity search for the question
query_embedding = np.array([embeddings[0]]).astype('float32')  # First item is the question
k = len(documents)  # Retrieve all options
distances, indices = index.search(query_embedding, k)

# Determine the closest (best match) option
best_match_index = indices[0][0]  # Closest match

# Print question, options, distances, and highlight the correct answer
print("\nQuestion:")
print(f"Q: {question}\n")

print("Options and Distances:")
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    option_text = documents[idx]  # Get the corresponding option text
    correct_marker = " ✅ Correct Answer" if i == 0 else ""  # Mark the closest match
    print(f"O{i+1}: {option_text} → Distance: {distance:.4f}{correct_marker}")

print()
# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot question (red)
q_x, q_y, q_z = reduced_embeddings[0]  # First point is the question
ax.scatter(q_x, q_y, q_z, color='red', label="Question", s=100)  # Bigger size for visibility
ax.text(q_x, q_y, q_z, "Q", fontsize=14, color='black', fontweight='bold')

# Plot options (blue)
doc_points = reduced_embeddings[1:]  # Remaining points are options
for i, (x, y, z) in enumerate(doc_points):
    ax.scatter(x, y, z, color='blue', label="Option" if i == 0 else "")
    ax.text(x, y, z, f"O{i+1}", fontsize=12)

    # Determine line color (Green for best match, Red for others)
    line_color = 'green' if i == best_match_index else 'red'

    # Draw similarity line from question to each option
    ax.plot([q_x, x], [q_y, y], [q_z, z], linestyle='--', color=line_color)

# Labels and title
ax.set_title("3D Visualization of Question and Option Embeddings")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.legend()
plt.show()

