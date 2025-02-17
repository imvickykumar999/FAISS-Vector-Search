import faiss
import numpy as np
import matplotlib
import sys  # Import sys to ensure full exit

matplotlib.use("TkAgg")  # Use Tkinter backend

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import Toplevel, Label, Entry, Button, Text
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Default question and options (Fixed Order)
default_question = "How many players are in a cricket team?"
default_options = [
    "There are 11 players in a cricket team.",
    "Basketball is played with 5 players per team.",
    "A kabaddi team consists of 7 players.",
    "A football team has 11 players.",
    "Hockey teams have 6 players on the ice."
]

# Global variables to hold question and options
current_question = default_question
current_options = default_options.copy()

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

# Function to open a pop-up window for user input (editable fields)
def edit_question_options():
    global current_question, current_options

    popup = Toplevel(root)
    popup.title("Edit Question and Options")

    # Question Label and Input
    Label(popup, text="Question:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    question_entry = Entry(popup, width=60)
    question_entry.grid(row=0, column=1, padx=5, pady=5)
    question_entry.insert(0, current_question)  # Pre-fill with current question

    # Options Label and Input Fields
    option_entries = []
    for i in range(len(current_options)):
        Label(popup, text=f"Option {i+1}:").grid(row=i+1, column=0, sticky="w", padx=5, pady=5)
        entry = Entry(popup, width=60)
        entry.grid(row=i+1, column=1, padx=5, pady=5)
        entry.insert(0, current_options[i])  # Pre-fill with current option
        option_entries.append(entry)

    # Function to save edited data
    def save_edits():
        global current_question, current_options
        current_question = question_entry.get()
        current_options = [entry.get() for entry in option_entries]

        if not current_question.strip():
            messagebox.showerror("Error", "Question cannot be empty!")
            return
        if any(not opt.strip() for opt in current_options):
            messagebox.showerror("Error", "Options cannot be empty!")
            return

        popup.destroy()  # Close the pop-up window
        generate_graph(current_question, current_options)  # Update graph

    # Save Button
    save_button = Button(popup, text="Save", command=save_edits, font=("Arial", 12))
    save_button.grid(row=len(current_options) + 1, column=0, columnspan=2, pady=10)

# Function to properly close Tkinter and exit program
def on_closing():
    root.destroy()  # Close Tkinter window
    sys.exit(0)  # Ensure full exit from CMD

# Tkinter GUI Setup
root = tk.Tk()
root.title("Question Similarity Visualization")

# Bind closing event to the exit function
root.protocol("WM_DELETE_WINDOW", on_closing)

label_instruction = tk.Label(root, text="Click 'Edit Question' to update", font=("Arial", 12))
label_instruction.pack(pady=10)

btn_edit = tk.Button(root, text="Edit Question", command=edit_question_options, font=("Arial", 12))
btn_edit.pack(pady=10)

label_result = tk.Label(root, text="", font=("Arial", 10), justify="left")
label_result.pack(pady=10)

frame = tk.Frame(root)
frame.pack()

generate_graph(current_question, current_options)

root.mainloop()

