from fastapi import APIRouter, UploadFile, File, HTTPException
from rag import rag_response, setup_rag
from model import Question_Metadata, Answer

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> Question_Metadata:
    # Handle the uploaded file (e.g., save it, process it, etc.)
    # For demonstration purposes, we'll just return a success message
    try:
        response = await setup_rag(file)
        document_id, filename, total_pages, total_chunks = response
        return Question_Metadata(
            message="File uploaded successfully.",
            document_id=document_id,
            filename=filename,
            total_pages=total_pages,
            total_chunks=total_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    

@router.get("/query/", response_model=Answer)
async def query_rag(query: str) -> Answer:
    try:
        response = await rag_response(query)
        return Answer(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))