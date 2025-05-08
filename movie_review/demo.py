import streamlit as st
import pandas as pd

from integration_langchain import integration_langchain

# review = st.text_input('리뷰', '이 영화 재밌어요')
url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
df = pd.read_csv(url, sep="\t")

# 전체 데이터 중 20개만 읽어옴.
data = df.iloc[:10].to_dict(orient='records')
# {키 : 값} => {'굳 ㅋ", {6270596	굳 ㅋ	1}
options = {item['document']: item for item in data}

# 다중 선택 리스트로 보여줌...

reviews = st.multiselect("리뷰", options.keys())
if st.button('submit'):
    selected_values = [options[doc] for doc in reviews]
    result = integration_langchain(selected_values)
    st.json(result)
