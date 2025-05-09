import json
import os
from typing import List

import dotenv
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from common.prompt_template import prompt_text_base
from image_base_work.inference_image import client
from text_base_work.download_text_data import get_text_data
from common.base_class import Output, QA
from openai import Client

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = Client(api_key=OPENAI_API_KEY)

# JSON 변환 파서 설정
output_parser = PydanticOutputParser(pydantic_object=Output)

def inference_json(info_data):
    prompt = prompt_text_base.format(
        format_instruction=output_parser.get_format_instructions(),
        info_text_data=info_data)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role":"system", "content":"You are a helpful assistant"
            },
            {
                "role":"user","content": prompt
            }
        ],
        temperature=0,
        response_format={"type":"json_object"}
    )

    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json

    # prompt = PromptTemplate(
    #     template=prompt_text_base,
    #     input_variables=['info_text_data'],
    #     partial_variables={"format_instruction":
    #        output_parser.get_format_instructions()}
    # )
    #
    # model = ChatOpenAI(
    #     temperature=0,
    #     api_key=OPENAI_API_KEY,
    #     model_name="gpt-4o-mini"
    # )
    #
    # chain = (prompt | model | output_parser)
    # output = chain.invoke({"info_text_data": info_data})
    # return output

if __name__ == '__main__':
    info_data = get_text_data()
    result = inference_json(info_data)
    print(json.dumps(result.dict(), indent=2, ensure_ascii=False))