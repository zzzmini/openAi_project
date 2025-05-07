import streamlit as st

from inference_json import inference_json
from predict_function_calling import predict_function_calling

review = st.text_input('리뷰', '이 영화 재밌어요')
if st.button('submit'):
    score = inference_json(review)
    keywords = predict_function_calling(review)
    st.write(score)
    st.write(keywords)
