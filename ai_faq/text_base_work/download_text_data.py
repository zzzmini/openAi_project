from langchain_community.document_loaders import AsyncHtmlLoader
from bs4 import BeautifulSoup
from langchain_community.document_transformers import Html2TextTransformer


def get_text_data():
    url="https://zzzmini.github.io/sample.html"

    # url 에서 HTML 긁어오기
    loader = AsyncHtmlLoader(url, verify_ssl=False)
    docs = loader.load()

    # HTML을 텍스트로 변환하기
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    content = docs_transformed[0].page_content
    return content

if __name__ == '__main__':
    print(get_text_data())