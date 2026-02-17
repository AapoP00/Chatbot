import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from rag import retrieve
from typing import List
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
Olet Bittipuisto.fi -sivuston asiantunteva asiakaspalvelubotti.

Vastaat käyttäjän kysymyksiin VAIN annettuun kontekstiin (lähteisiin) perustuen.

SÄÄNNÖT:
- Käytä vain annettua kontekstia vastauksen pohjana.
- Älä keksi tietoja, joita ei löydy kontekstista.
- Älä käytä yleistä tietämystä tai oletuksia.
- Jos vastausta ei löydy kontekstista, sano täsmälleen:
  "En löytänyt tietoa bittipuisto.fi-sivustolta."
- Älä mainitse sanaa "konteksti".
- Älä selitä miten haku toimii.
- Vastaa selkeästi ja tiiviisti suomeksi.
"""

@app.get("/health")
def health():
    return {"ok": True}

# Chat endpoint:
# 1. Retrieve top-k relevant chunks from vector index.
# 2. Build a context block for the LLM.
# 3. Aske the LLM to answer ONLY from that context.
# 4. Return the response.
@app.post("/chat")
def chat(req: ChatReq):
    user_message = (req.message or "").strip()
    if not user_message:
        return {"reply": "asd"}

    # Retrieve relevant chinks from the RAG index.
    hits = retrieve(client, user_message, k=6)

    # Build the context text and collect source URLs.
    context_parts: List[str] = []
    sources: List[str] = []

    for doc, meta in hits:
        src = meta.get("source")
        if src:
            sources.append(src)

        # Each chink is added with its source.
        context_parts.append(f"SISÄLTÖ:\n{doc}")

    # Deduplicate sources.
    dedup_sources = list(dict.fromkeys([s for s in sources if s]))

    # If none sources -> No call for Source list.
    if not dedup_sources:
        return {"reply": "En löytänyt tietoa bittipuisto.fi-sivustolta."}

    # Final context block.
    context = "\n\n---\n\n".join(context_parts) if context_parts else "EI LÄHTEITÄ."

    # RAG Instruction to ONLY answer from the provided context.
    # If cant answer, say "not found on bittipuisto.fi".
    rag_instruction = (
        "Vastaa VAIN annetun lähdekontekstin perusteella. "
        "Jos lähdekonteksti ei sisällä vastausta, sano: "
        "'En löytänyt tietoa bittipuisto.fi-sivustolta.' "
        "Älä keksi. Älä käytä ulkopuolisia lähteitä. "
        "Älä lisää vastauksen loppuun lähdeluetteloa tai URL-osoitteita."
    )

    # Call the model.
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "system", "content": rag_instruction},
            {"role": "system", "content": f"LÄHDEKONTEKSTI:\n{context}"},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    reply = (response.choices[0].message.content or "").strip()
    return {"reply": reply}