from pydantic import BaseModel

class Question_Metadata(BaseModel):
    message: str
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int

class Answer(BaseModel):
    answer: str
