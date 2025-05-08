from typing import List

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from prompt_template import prompt_integration_langchain
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# json 결과를 받을 클래스 선언
class Review(BaseModel):
    review_no: int
    evaluation: int
    positive_keywords: List[str]
    negative_keywords: List[str]

# Review들의 리스트와 요약정보를 담는 클래스
class OutputList(BaseModel):
    reviews: List[Review]
    summary: str

# OutputParser 선언
out_parser = PydanticOutputParser(pydantic_object=OutputList)

def integration_langchain(reviews):
    reviews = "\n".join([
            f"review_no:{review['id']}\tcontent:{review['document']}"
            for review in reviews
        ])

    # PromptTemplate
    prompt_langchain = PromptTemplate(
        template=prompt_integration_langchain,
        input_variables=['reviews'],
        partial_variables={"format_instructions":
                               out_parser.get_format_instructions()}
    )
    model = ChatOpenAI(
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        model_kwargs={"response_format":{"type":"json_object"}}
    )

    chain = (prompt_langchain | model | out_parser)
    output = chain.invoke({"reviews": reviews})
    return output

if __name__ == '__main__':
    reviews = [
      {"id": 1, "document": "뭐야 이 평점들은... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아"},
      {"id": 2, "document": "지루하지는 않은데 완전 막장임... 돈주고 보기에는..."},
      {"id": 3, "document": "3D만 아니었어도 별 다섯개 줬을텐데.. 왜 3D로 나와서 심기를 불편하게 하는지"},
      {"id": 4, "document": "음악이 주가 된, 최고의 음악영화"},
      {"id": 5, "document": "진정한 쓰레기"},
    ]
    print(integration_langchain(reviews))