from pydantic import BaseModel

class SpecifySearchResponse(BaseModel):
    specify_question: str
