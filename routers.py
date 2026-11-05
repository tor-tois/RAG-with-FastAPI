from fastapi import APIRouter, HTTPException
from rag import rag_response

router = APIRouter()

@router.get("/query/")
async def query_rag(query: str):
    try:
        response = await rag_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))