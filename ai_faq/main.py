from fastapi import FastAPI, Body
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from service.search_and_answer import search, generate_answer

app = FastAPI()

#CORS 설정
origins = {
    "http://localhost:3000"
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods={"*"},
    allow_headers={"*"}
)

class ReqeustBody(BaseModel):
    question:str

@app.post("/chatbot")
async def answer(body: ReqeustBody = Body()):
    qa = search(body.question)
    return generate_answer(qa, body.question)