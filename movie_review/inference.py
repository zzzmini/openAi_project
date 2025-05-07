from openai import Client
import prompt_template
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = Client(api_key=OPENAI_API_KEY)

def inference(review):
    # 프롬프트 생성
    prompt = prompt_template.prompt_question.format(review=review)

#     open_api 결과 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":"You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        # 답변의 유사도 : 0 ~ 2 (0 에 가까울 수록 정확한 답변)
        temperature=0
    )
    output = response.choices[0].message.content
    return output


if __name__ == '__main__':
    print(inference("돈 내기도 아까움. 절대 비추"))