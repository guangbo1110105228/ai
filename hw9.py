# 參考葉哲辰 取自專題測試部分
import openai
import time

openai.api_key = ""


def call_gpt(prompt):
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    end_time =  time.time()
    duration = end_time - start_time

    return response.choices[0].message['content'].strip(), duration 

prompt = "寫一段春天的描述。並且在結束要用標點符號結尾"
response_text, duration = call_gpt(prompt)
print(f"GPT-4的回應: {response_text}")
print(f"調用花費的時間: {duration:.2f} 秒")