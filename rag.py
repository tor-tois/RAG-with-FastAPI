from dotenv import load_dotenv
import os
import uuid
import shutil
from fastapi import UploadFile, File, HTTPException
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, 
    model="gemini-2.5-flash", 
    temperature=0.2
    )

upload_folder = "data"
upload_vector = "vectorstore"
retriever = None

os.makedirs(upload_folder, exist_ok=True)
os.makedirs(upload_vector, exist_ok=True)

# Function to load and process the PDF document
async def setup_rag(file: UploadFile = File(...)):

    # Check filename extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    
    content = await file.read()
    
    document_id = str(uuid.uuid4())

    file_path = os.path.join(upload_folder, f"{document_id}.pdf")

    # Save the uploaded file 
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    

    document_id = str(uuid.uuid4())

    # Load the PDF document
    pdfloader = PyMuPDFLoader(file_path)
    documents = pdfloader.load()

    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty or contains no readable text.")

    # Split documents into chunks
    spilitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", "."],
    chunk_size=1000,
    chunk_overlap=50
    )
    document_chunks = spilitter.split_documents(documents)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Create the vector store from the document chunks and embeddings
    db = FAISS.from_documents(
        documents=document_chunks, 
        embedding=embeddings
        )
    
    vectorstore_path = os.path.join(upload_vector, document_id)

    db.save_local(vectorstore_path)

    global retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return document_id, file.filename, len(documents), len(document_chunks)

async def rag_response(query: str):
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before submitting a query."
        )
    
    # Retrieve the relevant documents
    retriever_docs = await retriever.ainvoke(query)

    # Prepare the context for the LLM. Combine the query with the retrieved context
    context = "\n".join([doc.page_content for doc in retriever_docs])

    # Create a prompt for the LLM using the retrieved context and the user's query
    prompt = f"""
    Answer the question using only the following context.

    Context:
    {context}

    Question:
    {query}

    If the answer is not available in the context,
    say "I don't know."
    """

    # qa = RetrievalQA.from_chain_type(
    # llm=llm,
    # chain_type="stuff",
    # retriever=retriever
    # )

    # Generate a response using the LLM
    response = await llm.ainvoke(prompt)
    return response.content