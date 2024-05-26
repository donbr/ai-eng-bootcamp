from fastapi import FastAPI, File, Form, UploadFile
import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile
import os

app = FastAPI()

MODEL_NAME = "openai/whisper-medium"
BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), task: str = Form("transcribe")):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        inputs = ffmpeg_read(tmp_file.name, pipe.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    os.unlink(tmp_file.name)

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return {"text": text}