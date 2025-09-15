from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import tempfile

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ------------------------------------------------
# FastAPI app
# ------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hackathon: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded API key (for hackathon; use env var in prod!)
OPENAI_API_KEY = "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"

# Globals
vectordb = None
qa = None


# ------------------------------------------------
# Helper: Load documents
# ------------------------------------------------
def load_documents(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


# ------------------------------------------------
# Upload + index endpoint
# ------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectordb, qa

    # Save safely to temp folder
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    # Load + split
    docs = load_documents(temp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Build embeddings + vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(chunks, embeddings)

    # Conversational prompt
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a friendly and helpful AI assistant for employees.
Answer the question using the provided context.

- Use a conversational tone, like chatting with a colleague.
- If you don’t know, say so politely instead of guessing.
- Always include source citations at the end.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # 🔑 only store the final answer
    )

    # Conversational chain with sources
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY),
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        output_key="answer"
    )

    return {"message": f"File '{file.filename}' uploaded and processed successfully."}


# ------------------------------------------------
# Ask endpoint
# ------------------------------------------------
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    if qa is None:
        return {"answer": "No document uploaded yet. Please upload a file first."}

    result = qa.invoke(query.question)

    answer = result["answer"]

    sources = []
    for doc in result["source_documents"]:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "preview": doc.page_content[:150] + "..."
        })

    return {"answer": answer, "sources": sources}
