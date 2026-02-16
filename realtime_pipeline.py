import sounddevice as sd
import numpy as np
import whisper
import tempfile
import soundfile as sf

DURATION = 5
SAMPLE_RATE = 16000

print("Recording...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()

audio = audio.flatten()

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, audio, SAMPLE_RATE)
    temp_path = tmp.name

model = whisper.load_model("base")

print("Transcribing...")
result = model.transcribe(temp_path)

print("Transcript:", result["text"])
print("LLM Response: You said ->", result["text"])
