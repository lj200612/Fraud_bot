from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.text_classifier import TextClassifier
# from app.services.ner_service import NERService
# from app.services.similarity_service import SimilarityService
# from app.services.speech_service import SpeechService
# from app.services.knowledge_graph import KnowledgeGraph

router = APIRouter()

class TextRequest(BaseModel):
    text: str

# 初始化服务
text_classifier = TextClassifier()
# ner_service = NERService()
# similarity_service = SimilarityService()
# speech_service = SpeechService()
# knowledge_graph = KnowledgeGraph()

@router.post("/classify")
async def classify_text(request: TextRequest):
    """对输入文本进行诈骗类型分类"""
    try:
        result = text_classifier.classify(request.text)
        return {
            "type": result["type"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 其余接口全部注释
# @router.post("/ner")
# async def extract_entities(text: str):
#     ...
# @router.post("/similarity")
# async def calculate_similarity(text1: str, text2: str):
#     ...
# @router.post("/speech-to-text")
# async def speech_to_text(audio_file: bytes):
#     ...
# @router.post("/text-to-speech")
# async def text_to_speech(text: str):
#     ...
# @router.get("/knowledge/{fraud_type}")
# async def get_knowledge(fraud_type: str):
#     ... 