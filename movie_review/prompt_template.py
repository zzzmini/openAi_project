prompt_question = """
다음은 영화에 대한 리뷰입니다. 
영화에 대해 긍정적이면 1, 부정적이면 0으로 평가해 주세요.
...review
{review}
1 또는 0
...review"""

prompt_question_json = """
다음은 영화에 대한 리뷰입니다. 
영화에 대해 긍정적이면 1, 부정적이면 0으로 평가해 주세요.
아래 json 양식처럼 답변해 주세요.
{{
  "score": 0 or 1
}}
...review
{review}
...review"""

prompt_keyword ="""다음은 영화에 대한 리뷰입니다. 
리뷰에서 긍정적인 키워드, 부정적인 키워드를 추출해 주세요.
일반적인 답변으로 해 주세요
...review
{review}
...review"""


prompt_function_calling="""
다음은 영화에 대한 리뷰입니다.
리뷰에서 긍정적인 키워드, 부정적인 키워드를 추출해 주세요.
...review
{review}
...review
"""

prompt_summary="""
다음은 영화에 대한 리뷰들입니다.
리뷰 내용을 종합적으로 요약해 주세요.
아래 json 형식으로 응답해 주세요.
{{
  "summary":"이 영화는 ..."
}}
...reviews
{reviews}
...reviews
"""

prompt_summary_langchain="""
다음은 영화에 대한 리뷰들입니다.
리뷰 내용을 종합적으로 요약해 주세요.
{format_instruction}
...reviews
{reviews}
...reviews
"""
