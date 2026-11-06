from fastapi import APIRouter, HTTPException
from rag import rag_response
from model import Answer

router = APIRouter()

@router.get("/query/", response_model=Answer)
async def query_rag(query: str) -> Answer:
    try:
        response = await rag_response(query)
        return Answer(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))