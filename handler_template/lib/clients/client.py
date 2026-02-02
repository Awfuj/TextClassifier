import requests
from typing import List, Dict, Optional
from .dto import TextClassificationRequest, TextClassificationResponse, ClassificationResult


class BERTClassificationClient:
    """Класс клиента для взаимодействия с API классификации текста"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000"
        ):
        """
        Инициализация клиента
        
        Args:
            base_url: Базовый URL API сервера
        """
        self.base_url = base_url.rstrip('/')
        self.classify_endpoint = f"{self.base_url}/classify"
    
    def classify(
        self, 
        text: str, 
        top_k: int = 1,
        timeout: Optional[int] = 10
    ) -> TextClassificationResponse:
        """
        Классификация текста через API
        
        Args:
            text: Текст для классификации
            top_k: Количество топ классов для возврата
            timeout: Таймаут запроса в секундах
            
        Returns:
            TextClassificationResponse с результатами классификации
            
        Raises:
            requests.RequestException: При ошибке запроса
        """
        request_data = TextClassificationRequest(text=text, top_k=top_k)
        
        response = requests.post(
            self.classify_endpoint,
            json=request_data.dict(),
            timeout=timeout
        )
        response.raise_for_status()
        
        return TextClassificationResponse(**response.json())
    
    def classify_batch(
        self, 
        texts: List[str], 
        top_k: int = 1,
        timeout: Optional[int] = 30
    ) -> List[TextClassificationResponse]:
        """
        Классификация батча текстов через API
        
        Args:
            texts: Список текстов для классификации
            top_k: Количество топ классов для возврата
            timeout: Таймаут запроса в секундах
            
        Returns:
            Список TextClassificationResponse с результатами
            
        Raises:
            requests.RequestException: При ошибке запроса
        """
        results = []
        for text in texts:
            results.append(self.classify(text, top_k, timeout))
        return results
    
    def health_check(self) -> bool:
        """
        Проверка доступности API
        
        Returns:
            True если API доступен, False иначе
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

