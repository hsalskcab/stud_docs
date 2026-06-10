import time
from faster_whisper import WhisperModel
t=time.time()
try:
    model = WhisperModel("medium", device="cuda", compute_type="float16"); dev="cuda/float16"
except Exception as e:
    print("GPU failed:", repr(e)[:200]); 
    model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=20); dev="cpu/int8"
print("loaded model on", dev, "in", round(time.time()-t,1),"s")
t=time.time()
segs, info = model.transcribe("test90.wav", language="ru", beam_size=5, vad_filter=True)
for s in segs:
    print(f"[{s.start:6.1f}] {s.text.strip()}")
print("--- transcribed 90s in", round(time.time()-t,1),"s ---")
