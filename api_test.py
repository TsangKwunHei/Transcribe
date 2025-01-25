import os
from openai import OpenAI
from dotenv import load_dotenv
import shutil

# Load environment variables from .env file
load_dotenv()

# Check if the OpenAI API key is correctly loaded

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

prompt = "hi"
completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-3.5-turbo",
)

print(completion.choices[0].message.content)

