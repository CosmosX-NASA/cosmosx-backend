from sqlalchemy.orm import Session

from model import Figure

class FigureRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_research_id(self, research_id : int) -> list[Figure]:
        return (
            self.db.query(Figure)
                .filter(Figure.research_id == research_id)
                .all()
        )