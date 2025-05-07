from typing import List

from fastapi import FastAPI, Body
from openai import BaseModel

from inference_json import inference_json
from lang_chain_summary import lang_chain_summary
from predict_function_calling import predict_function_calling

app = FastAPI()

# 요청할 리뷰 구조를 정의하는 클래스
# 클래스명 뒤에 () -> 상속되는 클래스
class ReviewBody(BaseModel):
    review: str

# 리뷰 요청을 담을 클래스
class Summary(BaseModel):
    reviews: List[str]


@app.post("/evaluate")
async def evaluate_review(body: ReviewBody = Body()):
    return inference_json(body.review)

@app.post("/")
async def extract_review(body: ReviewBody=Body()):
    return predict_function_calling(body.review)

@app.post("/summary")
async def extract_summary(body: Summary = Body()):
    return lang_chain_summary(body.reviews)