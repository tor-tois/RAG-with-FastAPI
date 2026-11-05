from dotenv import load_dotenv
import os
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

# Function to load and process the PDF document
def setup_rag():
    # Load the PDF document
    pdfloader = PyMuPDFLoader("data/SAM.pdf")
    documents = pdfloader.load()

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
    db = FAISS.from_documents(documents=document_chunks, embedding=embeddings)

    # Create a retriever from the vector 
    retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
    )
    return retriever

async def rag_response(query: str):
    retriever = setup_rag()
    
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

