import cv2
import sys
import numpy as np
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage : python traitement_video.py video.mp4")
    sys.exit()

video_path = sys.argv[1]
video = cv2.VideoCapture(video_path)

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="Traitement vidéo")

# Compteurs
rouge = 0
jaune = 0
vert = 0
orange = 0
bleu = 0
noir = 0

colors = {
    "rouge": np.array([252, 0, 99]),
    "jaune": np.array([253, 254, 94]),
    "vert":  np.array([0, 254, 97]),
    "orange": np.array([250, 0, 0]),
    "bleu":  np.array([0, 0, 96])
}

while True:
    ret, frame = video.read()
    if not ret:
        break

    # conversion BGR en RGB si besoin
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MASQUES vectorises (ULTRA RAPIDE)
    mask_rouge = np.all(frame == colors["rouge"], axis=-1)
    mask_jaune = np.all(frame == colors["jaune"], axis=-1)
    mask_vert  = np.all(frame == colors["vert"], axis=-1)
    mask_orange = np.all(frame == colors["orange"], axis=-1)
    mask_bleu  = np.all(frame == colors["bleu"], axis=-1)

    # comptage
    rouge += np.sum(mask_rouge)
    jaune += np.sum(mask_jaune)
    vert  += np.sum(mask_vert)
    orange += np.sum(mask_orange)
    bleu   += np.sum(mask_bleu)

    # noir = tout le reste
    total_pixels = frame.shape[0] * frame.shape[1]
    noir += total_pixels - (
        np.sum(mask_rouge) +
        np.sum(mask_jaune) +
        np.sum(mask_vert) +
        np.sum(mask_orange) +
        np.sum(mask_bleu)
    )

    pbar.update(1)

pbar.close()
video.release()

print("Résultats :")
print("Rouge :", rouge)
print("Jaune :", jaune)
print("Vert  :", vert)
print("Orange:", orange)
print("Bleu  :", bleu)
print("Noir  :", noir)