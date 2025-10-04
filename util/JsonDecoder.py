import json
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class JsonDecoder:
    @staticmethod
    def decode(raw_text: str, model_class: Type[T]) -> Optional[T]:
        """
        raw_text를 JSON으로 파싱하고, 지정된 Pydantic 모델로 변환
        :param raw_text: JSON 문자열
        :param model_class: 변환할 Pydantic 모델 클래스
        :return: 모델 인스턴스 또는 None
        """
        try:
            data = json.loads(raw_text)
            return model_class(**data)
        except json.JSONDecodeError:
            print("JSON parsing failed:", raw_text)
            return None
        except TypeError as e:
            print("Type mismatch:", e, raw_text)
            return None