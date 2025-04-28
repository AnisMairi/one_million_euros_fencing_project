#!/usr/bin/env python
# coding: utf-8
"""
fencing_cutter.py
Découpe un match d'escrime en clips basés sur détection de mouvement et/ou LEDs de score.
"""

import cv2
import numpy as np
import argparse
import subprocess
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Découpe vidéo d'escrime en clips d'actions.")
    p.add_argument("--input",      required=True,  help="Chemin vers le fichier vidéo source")
    p.add_argument("--output_dir", default="videos/",  help="Dossier de sortie pour les clips")
    p.add_argument("--method",     default="hybrid", choices=["motion","led","hybrid"],
                   help="Méthode: motion, led, ou hybrid")
    p.add_argument("--motion_on",  type=int, default=5_000_000,
                   help="Seuil d’activation du mouvement (somme des diff.)")
    p.add_argument("--motion_off", type=int, default=2_000_000,
                   help="Seuil de fin du mouvement")
    p.add_argument("--min_on",     type=int, default=3,
                   help="Nb. de frames consécutives ≥ motion_on pour démarrer un clip")
    p.add_argument("--min_off",    type=int, default=5,
                   help="Nb. de frames consécutives ≤ motion_off pour clore un clip")
    p.add_argument("--pad_pre",    type=int, default=15,
                   help="Nb. de frames à rajouter avant le début détecté")
    p.add_argument("--pad_post",   type=int, default=20,
                   help="Nb. de frames à rajouter après la fin détectée")
    return p.parse_args()

def detect_motion_events(cap, threshold_on, threshold_off, min_on, min_off):
    events = []
    prev_gray = None
    consec_on = consec_off = 0
    in_event = False
    start_frame = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if in_event:
                events.append((start_frame, idx-1))
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray; idx+=1; continue

        diff = cv2.absdiff(gray, prev_gray)
        score = int(diff.sum())

        if not in_event:
            if score >= threshold_on:
                consec_on += 1
            else:
                consec_on = 0
            if consec_on >= min_on:
                in_event = True
                start_frame = idx - min_on + 1
                consec_off = 0
        else:
            if score <= threshold_off:
                consec_off += 1
            else:
                consec_off = 0
            if consec_off >= min_off:
                end_frame = idx - min_off
                events.append((start_frame, end_frame))
                in_event = False
                consec_on = 0

        prev_gray = gray
        idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return events

def validate_led_events(cap, events, templates, thresholds):
    """
    Ne garde que les events où l'une des LEDs scorable apparaît.
    templates = { 'white':img, 'green':img, 'red':img }
    thresholds = { 'white':7000, 'green':40000, 'red':40000 }
    boxes = [
      ('white', (337,348,234,250)),
      ('white', (337,348,390,406)),
      ('green', (330,334,380,500)),
      ('red',   (330,334,140,260)),
    ]
    """
    valid = []
    boxes = [
      ('white',(337,348,234,250)),
      ('white',(337,348,390,406)),
      ('green',(330,334,380,500)),
      ('red',  (330,334,140,260)),
    ]

    for (s, e) in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        hit = False
        for fidx in range(s, e+1):
            ret, frame = cap.read()
            if not ret:
                break
            for color, (y1,y2,x1,x2) in boxes:
                roi = frame[y1:y2, x1:x2]
                diff = int(np.sum(np.abs(roi.astype(int) - templates[color].astype(int))))
                if diff <= thresholds[color]:
                    valid.append((s,e))
                    hit = True
                    break
            if hit:
                break
        # réinitialise pour le prochain
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return valid

def extract_clips(input_path, output_dir, events, fps, pad_pre, pad_post):
    os.makedirs(output_dir, exist_ok=True)
    for i, (s,e) in enumerate(events):
        t0 = max(0, (s - pad_pre) / fps)
        t1 = (e + pad_post) / fps
        out = os.path.join(output_dir, f"clip_{i:03d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", f"{t0:.3f}",
            "-to", f"{t1:.3f}",
            "-c", "copy",
            out
        ]
        print(f"Extraction clip {i:03d}: {t0:.2f}s → {t1:.2f}s")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    args = parse_args()

    # Charge la vidéo
    #cap = cv2.VideoCapture(args.input)
    cap = cv2.VideoCapture("mon_match.mp4")
    if not cap.isOpened():
        print("Impossible d'ouvrir la vidéo.", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Charge les templates LED
    templates = {
      'white': cv2.imread("whitebox.png"),
      'green': cv2.imread("greenbox.png"),
      'red':   cv2.imread("redbox.png"),
    }
    thresholds = {'white':7000, 'green':40000, 'red':40000}

    for name,img in templates.items():
        if img is None:
            print(f"⚠️ Erreur : {name}box.png introuvable")


    # 1) Détection mouvement
    ev_motion = []
    if args.method in ("motion","hybrid"):
        print("→ Détection mouvement…")
        ev_motion = detect_motion_events(cap,
                                         args.motion_on,
                                         args.motion_off,
                                         args.min_on,
                                         args.min_off)
        print(f"  {len(ev_motion)} segments repérés par mouvement.")

    # 2) Validation LEDs
    ev_final = []
    if args.method == "led":
        # on considère comme events bruts tout le film
        ev_motion = [(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)]
    if args.method in ("led","hybrid"):
        print("→ Validation par LEDs…")
        ev_final = validate_led_events(cap, ev_motion, templates, thresholds)
        print(f"  {len(ev_final)} segments validés par LEDs.")
    else:
        ev_final = ev_motion

    cap.release()

    # 3) Extraction ffmpeg
    print("→ Extraction des clips…")
    extract_clips(args.input, args.output_dir, ev_final, fps,
                  args.pad_pre, args.pad_post)
    print("Terminé.")

if __name__ == "__main__":
    main()
