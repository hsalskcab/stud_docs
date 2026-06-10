import os, sys, time, glob, subprocess
import imageio_ffmpeg
from faster_whisper import WhisperModel

FF = imageio_ffmpeg.get_ffmpeg_exe()
LECT_DIR = "lect"
OUT_DIR = "transcripts"
os.makedirs(OUT_DIR, exist_ok=True)

def hm(s):
    s=int(s); return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}"

# термины курса -> подсказка распознавателю (уменьшает систематические ошибки)
PROMPT = ("Лекция по численным методам. Интерполяционный полином, параболический и "
          "кубический сплайн, факториал, узлы Чебышёва, перекличка, трёхдиагональная "
          "матрица, метод прогонки, разделённая разность, метод Эйлера, метод Рунге-Кутты, "
          "задача Коши, краевая задача, разностная схема, устойчивость, метод сеток, "
          "метод стрельбы, уравнение переноса, метод установления, гармоники, редукция.")

print("Загружаю модель medium на GPU...", flush=True)
t0=time.time()
model = WhisperModel("medium", device="cuda", compute_type="float32")
print(f"  модель загружена за {time.time()-t0:.0f}s", flush=True)

files = sorted(glob.glob(os.path.join(LECT_DIR, "*.mp4")),
               key=lambda p: int(''.join(c for c in os.path.basename(p).split('.')[0] if c.isdigit()) or 0))

for i, mp4 in enumerate(files, 1):
    base = os.path.splitext(os.path.basename(mp4))[0]
    out_txt = os.path.join(OUT_DIR, base + ".txt")
    if os.path.exists(out_txt):
        print(f"[{i}/{len(files)}] ПРОПУСК (уже есть): {base}", flush=True)
        continue
    wav = "/tmp/_lect_audio.wav"
    print(f"[{i}/{len(files)}] {base}", flush=True)
    t=time.time()
    subprocess.run([FF,"-y","-i",mp4,"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",wav],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    segs, info = model.transcribe(wav, language="ru", beam_size=5, vad_filter=True,
                                  initial_prompt=PROMPT)
    n=0
    with open(out_txt+".tmp","w") as f:
        for s in segs:
            f.write(f"[{hm(s.start)}] {s.text.strip()}\n"); n+=1
    os.replace(out_txt+".tmp", out_txt)
    os.remove(wav)
    print(f"    -> {n} сегментов, {time.time()-t:.0f}s", flush=True)

print("ВСЁ ГОТОВО", flush=True)
