import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Load BERT Model for Abstract Embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # Set BERT model to evaluation mode, so it does not learn

# Read abstracts from file
with open("abstracts.txt", "r") as f:
    abstracts = f.read().strip().split("\n\n")  # Split abstracts by double newline

# Embed the abstracts using BERT and its [CLS] token
def embed_input(text: str) -> np.ndarray:
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token embedding
    return cls_embedding.numpy().flatten()  # Flatten to 1D array

# Perform PCA via SVD to get principal components 
def PCA(embedded_matrix: np.ndarray, n_components: int = 10):
    centered_matrix = embedded_matrix - np.mean(embedded_matrix, axis=0)
    _, _, Vt = np.linalg.svd(centered_matrix)
    return Vt[:n_components]

# Top contributors by dot products and sorting
def find_top_contributors(principal_components: np.ndarray, embedded_matrix: np.ndarray, n_abstracts: int = 3):
    top_contributors = {}
    for i, pc in enumerate(principal_components):
        contributions = np.dot(embedded_matrix, pc) 
        top_indices = np.argsort(np.abs(contributions))[-n_abstracts:][::-1] 
        top_contributors[f"Axis {i+1}"] = top_indices
    return top_contributors

# axis name from user input
def give_axis_names(top_contributors, abstracts: np.ndarray, n_components: int = 10):
    axis_dict = {}
    for i in range(n_components):
        # Find the 3 most contributing papers to principal component
        abstract_1 = abstracts[top_contributors[f"Axis {i+1}"][0]]
        abstract_2 = abstracts[top_contributors[f"Axis {i+1}"][1]]
        abstract_3 = abstracts[top_contributors[f"Axis {i+1}"][2]]
        # Ask user for input given these abstracts, and add to dictionary
        axis_name = str(input(f"Give an appropriate axis name for the following abstracts, \n1. {abstract_1}\n2. {abstract_2}\n3. {abstract_3}\nInput: "))
        print("\n")
        axis_dict[f"Axis {i+1}"] = axis_name
    return axis_dict

principal_components = None

# Get the axis dict, bad naming as it is not main function, but was too late to change
def main():
    embedded_matrix = np.array([embed_input(abstract) for abstract in abstracts])
    global principal_components
    principal_components = PCA(embedded_matrix)
    top_contributors = find_top_contributors(principal_components, embedded_matrix)
    axis_dict = give_axis_names(top_contributors, abstracts)
    return axis_dict

if __name__ == "__main__":
    main()
