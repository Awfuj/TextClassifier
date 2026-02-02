from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import Optional

from .dto import TextClassificationRequest, TextClassificationResponse
from .models.bert_model import BERTClassifier


# Инициализация FastAPI приложения
app = FastAPI(
    title="RuBERT Sentiment Classification API",
    description="API для классификации тональности текста на основе rubert-tiny2",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация модели (можно вынести в отдельную функцию для ленивой загрузки)
bert_model: Optional[BERTClassifier] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация модели при запуске приложения"""
    global bert_model
    # Здесь можно указать конкретную модель
    # Например: "cointegrated/rubert-base-cased-nli-threeway" для русских текстов
    # или "distilbert-base-uncased-finetuned-sst-2-english" для английских
    model_name = "artifacts/model"
    bert_model = BERTClassifier(model_name=model_name)


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {"status": "healthy", "model_loaded": bert_model is not None}


@app.post("/classify", response_model=TextClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    if bert_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        result = bert_model.predict(request.text, top_k=request.top_k)

        return TextClassificationResponse(
            text=request.text,
            predictions=result["predictions"],
            processing_time=result["processing_time"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "BERT Binary Classification API",
        "docs": "/docs",
        "health": "/health"
    }

