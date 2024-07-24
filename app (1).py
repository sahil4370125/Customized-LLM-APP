import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Italian_Italian_200_Recipes_for_Authenti.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
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
                     "Remember you give one reciepe in short detail at a time."
                     "You gave concise and to the point  instructions."
                     "You use your vast culinary knowledge and the provided PDF to guide users through recipes and provide helpful information.")
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
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