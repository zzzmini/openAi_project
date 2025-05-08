# Embedding : 문장을 토큰 단위로 나누는 작업
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
import os

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from common.base_class import Output, QA
from langchain_core.output_parsers import PydanticOutputParser
from openai import Client

# PDF 변환
def attention_policy_chunking():
    # 훈련규정 읽어오기
    doc_policy = PyPDFLoader("../src/attendance_policy.pdf")
    # 전체 페이지를 통째로 split
    doc_policy_load = doc_policy.load()
    # print(doc_policy_load)

    # chunking 사이즈 규정
    policy_splitter = CharacterTextSplitter(
        separator=".\n",
        chunk_size=100,
        chunk_overlap=50,
        length_function=len
    )

    # 규정 PDF chunking하기
    att_chunk_doc = policy_splitter.split_documents(doc_policy_load)

    # 규정 확인해 보기
    # print(len(att_chunk_doc))
    # print(att_chunk_doc[1].page_content)

    # 텍스트 형식으로 되어 있는 출력인정일수 청킹하기
    with open("../src/attention_condition.txt","r", encoding="utf-8") as f:
        att_chunk_txt = f.read()

    att_text_doc = [Document(page_content=att_chunk_txt)]
    # print(att_text_doc)
    # chunking 사이즈 규정(구분자가 2개이기 때문에)
    policy_splitter = RecursiveCharacterTextSplitter(
        separators=["\nㆍ", "ㆍ"],
        chunk_size=100,
        chunk_overlap=30,
        # length_function=len
    )
    # 출석인정일수 청킹
    attention_table = policy_splitter.split_documents(att_text_doc)

    # 결과 확인
    # print(attention_table)

    # 두개 문서 청킹 결과를 하나 합치기
    raw_source = att_chunk_doc + attention_table
    return raw_source

if __name__ == '__main__':
    attention_policy_chunking()
