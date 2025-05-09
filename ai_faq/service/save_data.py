import os
import pickle

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from text_base_work.inference_text import inference_json
from text_base_work.download_text_data import get_text_data
from image_base_work.download_image_data import get_urls
from image_base_work.inference_image import inference_image_text
from pdf_base_work.generate_faq_json import generate_faq_json
from pdf_base_work.pdf_chunking_save import attention_policy_chunking

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    # 1. HTML -> Json 가져오기
    text_data = get_text_data()
    result_text_json = inference_json(text_data)

    # 2. Image -> Json 가져오기
    url_lists = get_urls()
    result_image_json = inference_image_text(url_lists)

    # 3. PDF, TEXT -> Json 가져오기
    file_data = attention_policy_chunking()
    result_file_json = generate_faq_json(file_data)

    # 세가지 방법으로 얻어온 결과를 합치기
    result = (result_text_json['qa_list']
             + result_image_json['qa_list']
             + result_file_json['qa_list'])

    # 결과 json 결과를 테스트에 사용하도록 pickle로 저장
    with open("../src/pickle/qas.pkl", "wb") as f:
        pickle.dump(result, f)

    # 질문만 Embeddings 처리하기 위해 질문만 따로 리스트로 생성
    result_question = [row['question'] for row in result]

    # local에 파일 형태로 DB를 생성
    # Vector DB에 저장 -> FAISS

    db = FAISS.from_texts(
        # 임베딩할 텍스트 리스트(질문) -> 키
        result_question,
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        metadatas=result # 검색 결과로 응답할 아이
    )

    # 저장하기
    db.save_local("../src/db/qas.index")

if __name__ == '__main__':
    main()