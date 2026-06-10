import time
from faster_whisper import WhisperModel
for ct in ["float32","int8_float32","int8"]:
    try:
        t=time.time(); m=WhisperModel("medium",device="cuda",compute_type=ct)
        segs,info=m.transcribe("test90.wav",language="ru",beam_size=5,vad_filter=True)
        n=sum(1 for _ in segs); 
        print(f"cuda/{ct}: OK, {n} segs, {round(time.time()-t,1)}s (incl load)")
        break
    except Exception as e:
        print(f"cuda/{ct}: FAIL {repr(e)[:120]}")
