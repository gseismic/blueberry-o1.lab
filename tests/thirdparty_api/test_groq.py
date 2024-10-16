import os
from groq import Groq

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

def test_groq_basic():
    # https://console.groq.com/docs/rate-limits#current-rate-limits-for-chat-completions
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
        model="llama3-8b-8192",
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    if 1:
        test_groq_basic()