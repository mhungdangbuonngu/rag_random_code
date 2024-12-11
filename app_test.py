import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import json
import ollama
import asyncio


# Load the pre-trained PhoBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('vinai/phobert-base')
model = BertModel.from_pretrained('vinai/phobert-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load precomputed embeddings
vectors = np.load('embeddings.npy')
vectors = vectors.astype(np.float32)

# FAISS index creation
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = f.readlines()
    # Remove any trailing newlines or extra spaces
    documents = [doc.strip() for doc in documents]
    return documents
# Function to convert a text into its embedding (PhoBERT)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=258)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU if available
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[0][0].cpu().numpy()  # Extract the embedding for the first token ([CLS])
    return embedding

documents = load_documents("output2(300).txt")

# FAISS retrieval function
def retrieve(info, k=5):
    k = int(k)
    info = np.array(info).reshape(1, -1).astype(np.float32)  # Ensure info is a numpy array
    distance, indices = index.search(info, k)  # Use the global 'index' object
    
    # Get the relevant documents based on indices
    relevant_docs = [documents[i] for i in indices[0]]
    return relevant_docs

# Define tools for function calling (for use with Ollama)
tools = [{
    "type": "function",
    "function": {
        "name": "retrive",
        "description": "Cung cấp những tài liệu/trích dẫn liên quan đến thông tin mà bạn tìm kiếm về luat phap",
        "parameters": {
            "type": "object",
            "properties": {
                "info": {
                    "type": "string",
                    "description": "Thông tin/lĩnh vực ve phap luat can tim kiem",
                },
                "k": {
                    'type': "constant",
                    'description': "so luong vector lay ra tu trong index",
                }
            },
            "required": ["info", "k"],
        },
    },
}]

# Function to run the model with the user input and call functions
async def run(model, user_input):
    client = ollama.AsyncClient()
    messages = [
        {
            'role': 'user',
            'content': user_input,
        }
    ]
    
    # Call the model using Ollama API
    response = await client.chat(
        model=model,
        messages=messages,
        tools=tools,
    )
    
    messages.append(response["message"])
    
    if not response['message'].get('tool_calls'):
        print("\nThe model didn't use the function. Its response was:")
        print(response["message"]['content'])
        return
    
    if response["message"].get("tool_calls"):
        available_functions = {
            "retrive": retrieve,
        }
        
        for tool in response["message"]["tool_calls"]:
            function_to_call = available_functions[tool["function"]["name"]]
            print(f"function to call: {function_to_call}")

            if function_to_call == retrieve:
                # Convert the string input 'info' into an embedding vector
                info = get_embedding(tool['function']['arguments']['info'])
                
                # Call the retrieve function with the embedded query
                function_response = function_to_call(
                    info,
                    tool['function']['arguments']['k'],
                )
                print(f"function response: {function_response}")
                
                # Combine the query and retrieved documents to create the full context
                input_text = user_input + " " + "\n".join(function_response)
                print(f"Generated input for Ollama: {input_text}")
            
            messages.append(
                {
                    "role": "tool",
                    "content": str(function_response),  # Ensure the response is in string format
                }
            )
            
            # Now use the context (query + retrieved docs) for generating the answer
            response = await client.chat(
                model=model,
                messages=messages,
                tools=[],
            )
            print("\nGenerated response from Ollama:")
            print(response["message"]['content'])

# Main loop to keep asking for user input
while True:
    user_input = input("\nPlease ask=> ")
    if not user_input:
        user_input = "What is the flight time from NYC to LAX?"
    if user_input.lower() == "exit":
        break

    asyncio.run(run("llama3.1", user_input))
