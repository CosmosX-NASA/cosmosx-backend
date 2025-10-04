from sqlalchemy.orm import Session
from model import Research


class ResearchRagRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_user_search(self, search: str, pageSize: int) -> list[Research]:
        pass
