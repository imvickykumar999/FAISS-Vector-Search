import faiss
import numpy as np
import matplotlib
import sys  # Import sys to force exit

matplotlib.use("TkAgg")  # Use Tkinter backend

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Default options (documents) - Fixed Order
default_options = [
    "There are 11 players in a cricket team.",
    "Basketball is played with 5 players per team.",
    "A kabaddi team consists of 7 players.",
    "A football team has 11 players.",
    "Hockey teams have 6 players on the ice."
]

# Function to generate embeddings, find similarity, and plot graph
def generate_graph(question, documents):
    all_texts = [question] + documents  # Question is the first item
    embeddings = model.encode(all_texts)
    embeddings = np.array(embeddings).astype('float32')

    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings[1:])  # Add only options (documents) to the FAISS index

    query_embedding = np.array([embeddings[0]]).astype('float32')
    k = len(documents)
    distances, indices = index.search(query_embedding, k)

    index_distance_map = {idx: dist for dist, idx in zip(distances[0], indices[0])}
    best_match_index = min(index_distance_map, key=index_distance_map.get)

    for widget in frame.winfo_children():
        widget.destroy()

    result_text = f"\nQuestion:\nQ: {question}\n\nOptions and Distances:\n"
    for i, option_text in enumerate(documents):
        distance = index_distance_map[i]
        correct_marker = " ✅ Correct Answer" if i == best_match_index else ""
        result_text += f"O{i+1}: {option_text} → Distance: {distance:.4f}{correct_marker}\n"

    label_result.config(text=result_text)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    q_x, q_y, q_z = reduced_embeddings[0]
    ax.scatter(q_x, q_y, q_z, color='red', label="Question", s=100)
    ax.text(q_x, q_y, q_z, "Q", fontsize=10, color='black', fontweight='bold')

    doc_points = reduced_embeddings[1:]
    for i, (x, y, z) in enumerate(doc_points):
        ax.scatter(x, y, z, color='blue', label="Option" if i == 0 else "")
        ax.text(x, y, z, f"O{i+1}", fontsize=10)

        line_color = 'green' if i == best_match_index else 'red'
        ax.plot([q_x, x], [q_y, y], [q_z, z], linestyle='--', color=line_color)

    ax.set_title("3D Visualization of Question and Option Embeddings")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to refresh and ask for a new question
def refresh():
    new_question = simpledialog.askstring("Input", "Enter a new question:")
    if new_question:
        generate_graph(new_question, default_options)
    else:
        messagebox.showerror("Error", "Question cannot be empty!")

# Function to properly close Tkinter and exit program
def on_closing():
    root.destroy()  # Close Tkinter window
    sys.exit(0)  # Ensure full exit from CMD

# Tkinter GUI Setup
root = tk.Tk()
root.title("Question Similarity Visualization")

# Bind closing event to the exit function
root.protocol("WM_DELETE_WINDOW", on_closing)

label_instruction = tk.Label(root, text="Click Refresh to Enter a New Question", font=("Arial", 12))
label_instruction.pack(pady=10)

btn_refresh = tk.Button(root, text="Refresh", command=refresh, font=("Arial", 12))
btn_refresh.pack(pady=10)

label_result = tk.Label(root, text="", font=("Arial", 10), justify="left")
label_result.pack(pady=10)

frame = tk.Frame(root)
frame.pack()

generate_graph("How many players are in a cricket team?", default_options)

root.mainloop()

