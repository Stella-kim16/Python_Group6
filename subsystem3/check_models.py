import os
from openai import OpenAI

# Initialize client with your KI:connect NRW credentials
client = OpenAI(
    base_url="https://chat.kiconnect.nrw/api/v1",
    api_key=os.getenv("KICONNECT_API_KEY")
)

try:
    print("Fetching available models from KI:connect NRW...")
    models = client.models.list()
    for m in models:
        print(f" - Model ID: {m.id}")
except Exception as e:
    print(f"Failed to list models: {e}")