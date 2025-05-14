from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(model="NousResearch/Llama-2-7b-chat-hf", token=HF_TOKEN)

response = client.text_generation(
    "What are the symptoms of Type 2 Diabetes?",
    max_new_tokens=200,
    temperature=0.5,
    top_p=0.95,
    do_sample=True
)

print(response)