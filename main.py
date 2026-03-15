from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

PORTFOLIO_CONTEXT = """You are an AI assistant on Hariharabalan M's portfolio website. Be friendly and concise (2-4 sentences).

NAME: Hariharabalan M | ROLE: AI/ML Engineer, Data Scientist, Full Stack Developer
LOCATION: Tirunelveli, Tamil Nadu | PHONE: +91 6384341504 | EMAIL: hariharabalannal5@gmail.com
LINKEDIN: linkedin.com/in/harihara-balan05
EDUCATION: B.E Computer Science, Francis Xavier Engineering College, CGPA 8.0 (2022-2026)
EXPERIENCE: AI/ML Intern at CSIR-4PI / National Aerospace Laboratories (Dec 2025 - Feb 2026)
SKILLS: Python, ML, Deep Learning, TensorFlow, Keras, Scikit-Learn, FastAPI, Flask, SQL, Docker, AWS, NLP, BERT, Computer Vision
PROJECTS:
1. ResumeAI Pro - Resume screening 87% accuracy, XGBoost + BERT
2. Engine Health Monitor - FastAPI real-time turbofan sensor visualization
3. Resource Booking System - Flask+SQLite with OTP auth for CSIR-4PI
Only answer about Hariharabalan."""

sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env")
    if req.session_id not in sessions:
        sessions[req.session_id] = []
    sessions[req.session_id].append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": PORTFOLIO_CONTEXT}] + sessions[req.session_id][-20:]
    try:
        r = requests.post(GROQ_URL, json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 512},
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=30)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        sessions[req.session_id].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "running"}