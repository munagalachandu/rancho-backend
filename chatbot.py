# rag_chatbot_mac.py

import os
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "combined_dataset_fixed.json"       # your JSON file
FAISS_INDEX_PATH = "faiss_index"


# ========== LOAD DATA ==========
def load_json_data(filepath: str):
    """Load JSON and return LangChain Documents"""
    with open(filepath, "r") as f:
        data = json.load(f)

    docs = []
    for idx, row in enumerate(data):
        content = row.get("text", json.dumps(row))  # fallback: full row
        docs.append(Document(page_content=content, metadata={"id": idx}))
    return docs


# ========== BUILD / LOAD FAISS ==========
def build_faiss_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


# ========== PROMPT ==========
prompt_template = """
You are "Sunny", a friendly, empathetic, and understanding virtual therapist. 
Your goal is to provide gentle, supportive, and thoughtful guidance to users seeking help. 
You are always sweet, encouraging, and patient, and you respond in a way that makes users feel heard, safe, and cared for.

You have access to a database of past therapy sessions in JSON format. Each record contains:
- "Context": the user's past statement or problem
- "Response": the previous therapeutic guidance or advice

When responding to a new user query, follow these rules:

1. Empathy First: Acknowledge the user's feelings before giving advice.
2. Friendly Tone: Use gentle, friendly, and encouraging language.
3. Practical Guidance: Offer small, actionable steps the user can take.
4. Non-Judgmental: Never criticize or judge the user's feelings or actions.
5. Reference Past Similar Cases: If retrieved context is similar, weave in relevant guidance naturally.
6. Encourage Reflection: Prompt the user to reflect on their feelings positively.
7. Keep it Safe: Never give medical or crisis advice; guide them to trained professionals if needed.

### User Query:
{question}

### Retrieved Therapy Records:
{context}

### Your Task:
Respond with empathy, encouragement, and practical guidance.
Use a warm, sweet, and supportive tone.
Integrate relevant insights from the retrieved records naturally.

Limit response to 2-5 paragraphs. Use comforting words like "I understand," "It's okay," "You're not alone," etc.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# ========== MAIN CHATBOT ==========
def get_qa_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


def run_chatbot():
    # Load or build FAISS
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        vectorstore = load_faiss_index()
    else:
        print("⚡ Building new FAISS index...")
        docs = load_json_data(DATA_PATH)
        vectorstore = build_faiss_index(docs)

    qa_chain = get_qa_chain(vectorstore)

    print("\n✅ Chatbot ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = qa_chain({"query": query})
        print("\nBot:", response["result"])


# ========== ENTRY ==========
if __name__ == "__main__":
    run_chatbot()