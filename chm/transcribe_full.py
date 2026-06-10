import time
from faster_whisper import WhisperModel
m=WhisperModel("medium",device="cuda",compute_type="float32")
t=time.time()
segs,info=m.transcribe("audio.wav",language="ru",beam_size=5,vad_filter=True)
def hm(s):
    s=int(s);return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}"
with open("transcript.txt","w") as f:
    for s in segs:
        line=f"[{hm(s.start)}] {s.text.strip()}"
        f.write(line+"\n"); f.flush()
print("DONE in",round(time.time()-t,1),"s")
