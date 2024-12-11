import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
# Load the pre-trained PhoBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('vinai/phobert-base')
model = BertModel.from_pretrained('vinai/phobert-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Function to get the embedding for a sentence
def get_embedding(text):
    # Tokenize the text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=258)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU if available
    print("Tokenization complete. Feeding input into model...")
    # Get the output of the model (hidden states)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # We extract the embedding from the last hidden state of the [CLS] token
    embedding = outputs.last_hidden_state[0][0].cpu().numpy()  # Extracting the embedding for the first token ([CLS])
    return embedding

# Function to read lines from the text file and generate embeddings
def process_file_and_embed(file_path):
    embeddings = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read each line from the file
        for line in tqdm(file,desc='Processing text',unit='line'):
            line = line.strip()  # Remove any extra whitespace or newlines
            embedding = get_embedding(line)
            embeddings.append(embedding)
    
    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(embeddings)
    return embeddings_array

# File path for your text file
file_path = 'output2.txt'  # Replace with the actual file path

# Get embeddings for all lines in the file
embeddings_array = process_file_and_embed(file_path)

# Save the embeddings to a .npy file
np.save('embeddings.npy', embeddings_array)

print("Embeddings generated and saved to 'embeddings.npy'.")
