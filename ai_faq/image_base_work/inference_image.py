import json
from http.client import responses

from langchain_core.output_parsers import PydanticOutputParser
from openai import Client
import os

from common.prompt_template import prompt_image_base
from image_base_work.download_image_data import get_urls
from common.base_class import Output, QA

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = Client(api_key=OPENAI_API_KEY)

# JSON 변환 파서 설정
output_parser = PydanticOutputParser(pydantic_object=Output)

def inference_image_text(url_list):
    prompt = prompt_image_base.format(
        format_instruction=output_parser.get_format_instructions())
    content = [
        {"type": "text", "text": prompt}
    ]
    for url in url_list[:5]:
        content.append({
            "type":"image_url",
            "image_url":{
                "url": url
            }
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user", "content": content}
        ],
        max_tokens=1000,
        response_format={"type":"json_object"}
    )
    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json

if __name__ == '__main__':
    url_list = get_urls()
    result = inference_image_text(url_list)
    print(json.dumps(result, indent=2, ensure_ascii=False))