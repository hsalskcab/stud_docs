#!/usr/bin/env python3
"""Идемпотентная пост-обработка транскриптов: чинит систематические ошибки ASR
в терминах численных методов. Безопасно запускать повторно."""
import sys, glob, os, re

# (паттерн, замена). Регистронезависимо по первой букве; сохраняем заглавную.
RAW = [
    (r"перекричк",      "перекличк"),     # перекличка
    (r"проболическ",    "параболическ"),
    (r"праболическ",    "параболическ"),
    (r"интерпульционн", "интерполяционн"),
    (r"интерполюционн", "интерполяционн"),
    (r"интерпульционал","интерполяционал"),
    (r"сплавен",        "сплайн"),         # сплайн/сплайнами/...
    (r"сплавён",        "сплайн"),
    (r"поляном",        "полином"),
    (r"полянома",       "полинома"),
    (r"бактериал",      "факториал"),
    (r"агибраическ",    "алгебраическ"),
    (r"трехдиагональн", "трёхдиагональн"),
]

def make_repl(pat, rep):
    # вариант с заглавной первой буквой
    cap_pat = pat[0].upper()+pat[1:]
    cap_rep = rep[0].upper()+rep[1:]
    return [(re.compile(cap_pat), cap_rep), (re.compile(pat), rep)]

REPL = [r for p,s in RAW for r in make_repl(p,s)]

def fix(text):
    n=0
    for rx, rep in REPL:
        text, c = rx.subn(rep, text); n+=c
    return text, n

def main(files):
    for fp in files:
        with open(fp, encoding="utf-8") as f: t=f.read()
        nt, n = fix(t)
        if n:
            with open(fp,"w",encoding="utf-8") as f: f.write(nt)
        print(f"{os.path.basename(fp)}: {n} замен")

if __name__=="__main__":
    files = sys.argv[1:] or sorted(glob.glob("transcripts/*.txt"))
    main(files)
