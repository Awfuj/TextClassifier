from pydantic import BaseModel, Field
from typing import List, Optional


class TextClassificationRequest(BaseModel):
    """Модель для входных данных запроса бинарной классификации текста"""
    text: str = Field(..., description="Текст для классификации", min_length=1)
    top_k: int = Field(3, ge=1, le=10)

class ClassificationResult(BaseModel):
    """Модель для одного результата классификации"""
    label: str = Field(..., description="Название класса")
    score: float = Field(..., description="Уверенность модели (0-1)", ge=0.0, le=1.0)


class TextClassificationResponse(BaseModel):
    """Модель для выходных данных ответа классификации"""
    text: str = Field(..., description="Исходный текст")
    predictions: List[ClassificationResult] = Field(..., description="Список предсказаний")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")

