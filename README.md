# Customized-LLM-APP

## Overview

This code is designed to create a chatbot that functions as an Italian chef, providing users with recipes and culinary advice. The chatbot leverages a PDF of Italian recipes, a SentenceTransformer model for semantic search, and the Hugging Face API for generating responses. The application is built using Gradio for the user interface.

## How It Works

### Components

1. **Gradio**: A Python library to create user interfaces for machine learning models.
2. **Hugging Face API**: Used for generating responses from a language model.
3. **PyMuPDF (fitz)**: A library to extract text from PDF files.
4. **Sentence Transformers**: A library to generate sentence embeddings.
5. **Faiss**: A library for efficient similarity search and clustering of dense vectors.

### Functionality

#### 1. Initializing the App

The `MyApp` class is initialized, setting up the application's state, loading the PDF, and building the vector database for semantic search.

#### 2. Loading the PDF

The `load_pdf` method extracts text from each page of the provided PDF file and stores it in the `documents` list.

#### 3. Building the Vector Database

The `build_vector_db` method creates embeddings for the extracted text using the SentenceTransformer model. These embeddings are then added to a Faiss index for efficient similarity search.

#### 4. Searching Documents

The `search_documents` method takes a user query, generates an embedding for it, and uses the Faiss index to find and return the most relevant documents.

#### 5. Responding to User Queries

The `respond` function handles user interactions. It builds a conversation history, retrieves relevant documents based on the user query, and generates a response using the Hugging Face API.

### User Interface

The Gradio interface consists of a chatbot that provides examples of questions users can ask. It also includes a disclaimer noting that the chatbot is based on a publicly available Italian cuisine book and that the use of the chatbot is at the user's own responsibility.

### Code Breakdown

```python
import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Italian_Italian_200_Recipes_for_Authenti.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = ("You are a knowledgeable and experienced Italian Chef, You always guide user about the dish or recipe in a brief manner with proper cooking methodology."
                     "Remember you give one recipe in short detail at a time."
                     "You gave concise and to the point instructions."
                     "You use your vast culinary knowledge and the provided PDF to guide users through recipes and provide helpful information.")
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=1500,
        stream=True,
        temperature=0.8,
        top_p=0.9,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on a publicly available Italian cuisine book. "
        "We are not professional chefs, and the use of this chatbot is at your own responsibility.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["How to make Italian Gelato?"],
            ["How to make grilled garlic bread?"],
            ["How to make creamy tomato-basil soup?"],
            ["How to bake cookies that stay soft for days?"],
            ["How to make gluten-free cookies?"],
            ["How to make Polenta pudding?"]
        ],
        title='The_Italian_Chef 👨‍🍳'
    )

if __name__ == "__main__":
    demo.launch()
```

### Explanation

1. **Initialization**:
   - The `MyApp` class is initialized, setting up the necessary components such as documents, embeddings, and index.
   - The `load_pdf` method extracts text from the PDF file and stores it in the `documents` list.
   - The `build_vector_db` method generates embeddings for the text and adds them to a Faiss index.

2. **Document Search**:
   - The `search_documents` method takes a user query, generates an embedding for it, and searches the Faiss index for the most relevant documents.

3. **User Interaction**:
   - The `respond` function handles the conversation, retrieves relevant documents, and generates responses using the Hugging Face API.

4. **Gradio Interface**:
   - The Gradio interface is set up to provide a chatbot interface with example queries and a disclaimer.

By following these steps, you can create a Retrieval-Augmented Generation (RAG) chatbot that enhances the capabilities of a language model by incorporating external knowledge from a PDF of Italian recipes.
