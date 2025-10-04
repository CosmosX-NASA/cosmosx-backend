import json
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class JsonDecoder:

    @staticmethod
    def decode(raw_text: str, model_class: Type[T]) -> Optional[T]:
        try:
            data = json.loads(raw_text)
            return model_class(**data)
        except json.JSONDecodeError:
            print("JSON parsing failed:", raw_text)
            return None
        except TypeError as e:
            print("Type mismatch:", e, raw_text)
            return None