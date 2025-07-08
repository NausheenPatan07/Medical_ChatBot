from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load PDF
loader = PyPDFLoader("data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf")  # 👈 put your PDF inside `data/` folder
documents = loader.load()

# Step 2: Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Step 3: Generate embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Build FAISS DB
db = FAISS.from_documents(docs, embedding_model)

# Step 5: Save to local path
db.save_local("vectorstore/db_faiss")

print("✅ Vectorstore created successfully!")
