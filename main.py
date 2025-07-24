# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from colorthief import ColorThief
import io
import speech_recognition as sr
import pyttsx3

app = FastAPI()

# Allow your frontend (React) to talk with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, use specific origins!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Color detection backend running!"}

@app.post("/detect-dominant-color/")
async def detect_dominant_color(file: UploadFile = File(...)):
    contents = await file.read()
    color_thief = ColorThief(io.BytesIO(contents))
    dominant_color = color_thief.get_color(quality=1)
    return {"dominant_color": dominant_color}

@app.post("/voice-query/")
async def voice_query(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file.file)
    with audio_file as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return {"recognized_text": text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/text-to-speech/")
async def text_to_speech(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return {"status": "Spoken"}

# Later: Add video upload and color/emotion detection
