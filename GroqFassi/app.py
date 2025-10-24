from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import faiss
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from groq import Groq
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()
# ------------------------------------------------

app = Flask(__name__)
SCRAPED_DATA_FILE = "static/scraped_data.json"

# Now GROQ_API_KEY will be successfully loaded if present in .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the key was loaded before initializing the client (good practice)
if not GROQ_API_KEY:
    # Use a print statement instead of raise for better Flask debugging in development
    print("Warning: GROQ_API_KEY not found. API functions will likely fail.")
    client = None # Initialize client to None if key is missing
else:
    client = Groq(api_key=GROQ_API_KEY)


# Step 1: Extract URLs from the Sitemap
sitemap_url = 'https://blogforge.pythonanywhere.com/sitemap.xml'

# Step 2: Fetch Meta Descriptions
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def fetch_meta_descriptions():
    """
    Fetches blog post descriptions from the sitemap URLs.
    Caches the results to a JSON file to avoid repeated scraping.
    """
    if os.path.exists(SCRAPED_DATA_FILE):
        print(f"Loading data from cached file: {SCRAPED_DATA_FILE}")
        with open(SCRAPED_DATA_FILE, "r") as file:
            data = json.load(file) # Load into a variable first
            print("--- URLs loaded from cache ---")
            for url in data.keys(): # Iterate and print URLs
                print(f"Cached URL: {url}")
            print("------------------------------")
            return data # Then return the data
    
    print(f"Scraping data from sitemap: {sitemap_url}")
    response = requests.get(sitemap_url)
    sitemap_xml = response.content
    root = ET.fromstring(sitemap_xml)
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = [url.find('ns:loc', namespace).text for url in root.findall('ns:url', namespace)]
    
    data = {}
    for url in urls:
        try:
            page_response = requests.get(url, headers=headers)
            if page_response.status_code == 200:
                soup = BeautifulSoup(page_response.text, 'html.parser')
                # Try to find the main blog content first (as in original script)
                blog_details = soup.find(class_="blog-details")
                
                if blog_details:
                    data[url] = blog_details.get_text(strip=True)
                else:
                    # Fallback to meta description
                    description_tag = soup.find('meta', attrs={'name': 'description'})
                    data[url] = description_tag['content'] if description_tag and 'content' in description_tag.attrs else 'No meta description found'
            else:
                data[url] = f'Error: {page_response.status_code}'
            
        except Exception as e:
            data[url] = f'Error fetching {url}: {str(e)}'
    
    # Ensure static directory exists before writing cache
    os.makedirs('static', exist_ok=True)
    with open(SCRAPED_DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)
    
    return data

default_options = list(set(fetch_meta_descriptions().values()))

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer model loaded.")

# Function to generate embeddings, find similarity, and plot graph
def generate_graph(question, documents):
    """
    Encodes the question and documents, performs PCA for 3D reduction,
    finds the best matching document using FAISS (L2 distance), and plots the results.
    """
    all_texts = [question] + documents
    embeddings = model.encode(all_texts)
    embeddings = np.array(embeddings).astype('float32')

    # Step 1: PCA for Visualization
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Step 2: FAISS Index for Similarity Search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings[1:]) # Add document embeddings

    # Step 3: Search
    query_embedding = np.array([embeddings[0]]).astype('float32')
    distances, indices = index.search(query_embedding, len(documents))
    
    # Identify the best match (minimum distance in L2 index)
    index_distance_map = {idx: dist for dist, idx in zip(distances[0], indices[0])}
    best_match_index_faiss = min(index_distance_map, key=index_distance_map.get)
    best_match_document = documents[best_match_index_faiss]

    # Step 4: Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Question (Q)
    q_x, q_y, q_z = reduced_embeddings[0]
    ax.scatter(q_x, q_y, q_z, color='#EF4444', label="Question (Q)", s=150, edgecolors='black', marker='D')
    ax.text(q_x, q_y, q_z, "Q", fontsize=12, color='black', fontweight='bold', ha='center', va='center')
    
    # Plot Document Options (O1, O2, ...)
    doc_points = reduced_embeddings[1:]
    for i, (x, y, z) in enumerate(doc_points):
        is_best_match = i == best_match_index_faiss
        
        ax.scatter(x, y, z, color='#3B82F6', s=80, alpha=0.7)
        ax.text(x, y, z, f"O{i+1}", fontsize=10)
        
        # Draw line from Question to Document
        line_color = '#10B981' if is_best_match else '#FBBF24'
        ax.plot([q_x, x], [q_y, y], [q_z, z], linestyle=':', color=line_color, alpha=0.6)
        
        # Highlight the best match point with a distinct marker/color
        if is_best_match:
            ax.scatter(x, y, z, color='#10B981', s=120, edgecolors='black', marker='*')

    ax.set_title("3D PCA Visualization: Question vs. Blog Content Embeddings", fontsize=14)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    
    # Create a small legend for colors/markers
    ax.legend(loc='lower left', handles=[
        plt.Line2D([0], [0], marker='D', color='w', label='Question (Q)', markerfacecolor='#EF4444', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Options (O)', markerfacecolor='#3B82F6', markersize=10),
        plt.Line2D([0], [0], linestyle=':', color='#10B981', label='Best Match Link'),
        plt.Line2D([0], [0], linestyle=':', color='#FBBF24', label='Other Links'),
    ])

    # Save and close the plot
    graph_path = "static/graph.png"
    plt.savefig(graph_path)
    plt.close(fig) # Ensure figure is closed to free memory
    
    return best_match_document, graph_path

def generate_reply(message_text):
    """
    Uses Groq's Llama 3.1 8B Instant model to generate a concise reply 
    based on the best-matched document content.
    """
    if client is None:
        return "API Key Missing: Cannot generate reply without GROQ_API_KEY."

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a concise blog summarizer. Respond only to the user's prompt based on the provided text, keeping the reply under 50 words."},
                {"role": "user", "content": message_text}
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        return f"Sorry, I'm having trouble processing your request due to an API error: {str(e)}"

# --- API ROUTE FOR AJAX CALLS ---
@app.route('/api/search', methods=['POST'])
def api_search():
    """Handles the API request for vector search and Groq generation."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "Question field is required"}), 400

    try:
        # 1. Find the most relevant document and generate the visualization
        best_match_content, graph_path = generate_graph(question, default_options)
        
        # --- NEW DEBUG PRINT STATEMENT ---
        print("\n--- API SEARCH DEBUG ---")
        print(f"User Question: {question}")
        print(f"Best Matched Content Used for RAG: {best_match_content[:150]}...") # Print first 150 chars
        print("------------------------\n")
        # ---------------------------------

        # 2. Use the relevant content to generate a Groq reply
        prompt = f'User Question: "{question}" \n\n Based on the following relevant content, write a helpful and short reply (under 50 words): \n\n Content: "{best_match_content}"'
        correct_answer = generate_reply(prompt)
        
        # Return results as JSON
        return jsonify({
            "answer": correct_answer,
            "graph_url": f"/{graph_path}", # Use /static/graph.png
            "question": question
        })
        
    except Exception as e:
        print(f"Error during API processing: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- MAIN ROUTE FOR SERVING HTML ---
@app.route('/', methods=['GET'])
def index():
    """Main route for serving the HTML template."""
    # Ensure static directory exists for saving the graph
    os.makedirs('static', exist_ok=True)
    return render_template('index.html')


if __name__ == '__main__':
    print("Application starting. Ensure you have a 'templates/index.html' file and a '.env' file with GROQ_API_KEY.")
    app.run(debug=True)
