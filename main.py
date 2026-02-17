import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fetch allowed domains (prevents other domains from calling this API)
allowed = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

# Add CORS-middleware
# Required for the site to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define frontend data type
class ChatReq(BaseModel):
    message: str

# Define bots behaviour
SYSTEM_PROMPT = """
You are friendly customer service bot.
You only speak english.
Keep your answers clear and professional.
"""

@app.get("/health")
def health():
    return {"ok": True}

# Create POST endpoint, that the JS calls.
@app.post("/chat")
def chat(req: ChatReq):
    # Send the message to OpenAI
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ],
        temperature=0.4,
    )

    # Returns the reply as JSON
    return {
        "reply": response.choices[0].message.content
    }