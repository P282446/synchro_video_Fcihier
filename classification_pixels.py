#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:50:35 2026

@author: seydou
"""

import os
#import pandas as pd
import cv2
#import sys
import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt

def classification(video_path) :
    
    #video_path = sys.argv[1]
    
    cap = video_path
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Segmentation")
    
    # Couleurs (ATTENTION OpenCV = BGR)
    colors = {
        "rouge": np.array([99, 0, 252]),   # rouge
        "jaune": np.array([94, 254, 253]), # jaune
        "vert": np.array([97, 254, 0]),   # vert
        "orange": np.array([0, 0, 250]),    # orange
        "bleu": np.array([96, 0, 0])      # bleu
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # image label (par pixel)
        label = np.full(frame.shape[:2], "noir", dtype=object)  # 5 = noir/autre
    
        # segmentation vectorisée
        for k, color in colors.items():
            mask = np.all(frame == color, axis=-1)
            label[mask] = k
        break
        # label contient maintenant la zone d'appartenance de chaque pixel
    
        pbar.update(1)
    
    pbar.close()
    cap.release()
    """
    print(label)
    plt.imshow(label, cmap="jet")
    plt.title("Segmentation des pixels")
    plt.show()
    """
    return label



def classification_with_coor(video_path, coor):

    # Copie du DataFrame afin de ne pas modifier les données d'origine
    sync = coor.copy()

    # Les coordonnées (0,0) correspondent à une perte de suivi de l'eye-tracker.
    # Elles sont remplacées par NaN afin de ne pas être classées dans la zone "Autre".
    mask = (sync["Lft X Pos"] == 0) & (sync["Lft Y Pos"] == 0)
    sync.loc[mask, ["Lft X Pos", "Lft Y Pos"]] = np.nan

    coor = sync.to_numpy()
    n = len(coor)
    zones = np.zeros(n, dtype=np.uint8)

    xx = coor[:, 0]
    yy = coor[:, 1]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Synchronisation")

    # Couleurs (ATTENTION OpenCV = BGR)
    colors = {
        1: np.array([97, 254, 0]),    # vert  -> Yeux
        2: np.array([99, 0, 252]),    # rose  -> Nez et Bouche
        3: np.array([94, 254, 253]),  # jaune -> Visage
        4: np.array([0, 0, 250]),     # rouge -> Haut du corps
        5: np.array([96, 0, 0]),      # bleu  -> Reste de la tête
    }

    TOLERANCE = 10  # marge d'erreur pour absorber les artefacts de compression
    MIN_COLORED_PIXELS = 100  # seuil minimal de pixels colorés pour considérer la frame comme valide

    temp = -3  # démarre à -3 pour que la 1re frame utilise range(0,3)
    nb_frames_invalides = 0

    while True:

        temp = temp + 3

        ret, frame = cap.read()

        if not ret:
            break

        # image label (par pixel), uint8 car les codes vont de 0 à 5
        label = np.zeros(frame.shape[:2], dtype=np.uint8)  # 0 = noir/autre

        # Conversion une seule fois en int16 pour éviter
        # un dépassement de capacité lors du calcul des différences de couleurs.
        frame16 = frame.astype(np.int16)

        # segmentation vectorisée avec tolérance sur la couleur
        total_colored_pixels = 0
        for k, color in colors.items():
            diff = np.abs(frame16 - color)
            color_mask = np.all(diff <= TOLERANCE, axis=-1)
            label[color_mask] = k
            total_colored_pixels += np.count_nonzero(color_mask)

        # Détection d'une frame mal segmentée : aucune (ou trop peu de)
        # zone colorée détectée -> probablement une frame noir et blanc / corrompue
        frame_invalide = total_colored_pixels < MIN_COLORED_PIXELS
        if frame_invalide:
            nb_frames_invalides += 1

        # label contient maintenant la zone d'appartenance de chaque pixel

        # min(temp+3, n) évite de dépasser la taille du tableau.
        for i in range(temp, min(temp + 3, n)):

            # Si l'eye-tracker n'a pas mesuré la position du regard,
            # les coordonnées sont NaN. On attribue alors un code spécifique
            # (255 = perte de suivi) afin de ne pas les confondre avec la zone "Autre".
            if np.isnan(xx[i]) or np.isnan(yy[i]):
                zones[i] = 255
                continue

            # Frame mal segmentée (noir et blanc / corrompue) : on ignore ce point
            # plutôt que de le classer à tort dans la zone "Autre".
            # 254 = frame invalide (segmentation défaillante)
            if frame_invalide:
                zones[i] = 254
                continue

            x = int(xx[i])
            y = int(yy[i])

            # sécurité : coordonnées hors image (regard perdu, clignement...)
            if 0 <= y < label.shape[0] and 0 <= x < label.shape[1]:
                zones[i] = label[y, x]
            else:
                # Les coordonnées hors image sont également
                # considérées comme une perte de suivi.
                zones[i] = 255

        pbar.update(1)

    cap.release()
    pbar.close()

    if nb_frames_invalides > 0:
        print(f"⚠️  {nb_frames_invalides} frame(s) mal segmentée(s) détectée(s) et ignorée(s) "
              f"sur {total} frames au total.")

    # Ajout de la colonne contenant la zone associée à chaque point de regard.
    sync["zones"] = zones

    return sync

def classification_with_coor3(video_path, coor):

    # Copie du DataFrame afin de ne pas modifier les données d'origine
    sync = coor.copy()

    # Les coordonnées (0,0) correspondent à une perte de suivi de l'eye-tracker.
    # Elles sont remplacées par NaN afin de ne pas être classées dans la zone "Autre".
    mask = (sync["Lft X Pos"] == 0) & (sync["Lft Y Pos"] == 0)
    sync.loc[mask, ["Lft X Pos", "Lft Y Pos"]] = np.nan

    coor = sync.to_numpy()
    n = len(coor)
    zones = np.zeros(n, dtype=np.uint8)

    xx = coor[:, 0]
    yy = coor[:, 1]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Synchronisation")

    # Couleurs (ATTENTION OpenCV = BGR)
    colors = {
        1: np.array([97, 254, 0]),    # vert  -> Yeux
        2: np.array([99, 0, 252]),    # rose  -> Nez et Bouche
        3: np.array([94, 254, 253]),  # jaune -> Visage
        4: np.array([0, 0, 250]),     # rouge -> Haut du corps
        5: np.array([96, 0, 0]),      # bleu  -> Reste de la tête
    }

    TOLERANCE = 10  # marge d'erreur pour absorber les artefacts de compression

    temp = -3  # démarre à -3 pour que la 1re frame utilise range(0,3)

    while True:

        temp = temp + 3

        ret, frame = cap.read()

        if not ret:
            break

        # image label (par pixel), uint8 car les codes vont de 0 à 5
        label = np.zeros(frame.shape[:2], dtype=np.uint8)  # 0 = noir/autre

        # Amélioration : conversion une seule fois en int16 pour éviter
        # un dépassement de capacité lors du calcul des différences de couleurs.
        frame16 = frame.astype(np.int16)

        # segmentation vectorisée avec tolérance sur la couleur
        for k, color in colors.items():
            diff = np.abs(frame16 - color)
            mask = np.all(diff <= TOLERANCE, axis=-1)
            label[mask] = k

        # label contient maintenant la zone d'appartenance de chaque pixel

        # Amélioration : min(temp+3, n) évite de dépasser la taille du tableau.
        for i in range(temp, min(temp + 3, n)):

            # Amélioration : si l'eye-tracker n'a pas mesuré la position du regard,
            # les coordonnées sont NaN. On attribue alors un code spécifique
            # (255 = perte de suivi) afin de ne pas les confondre avec la zone "Autre".
            if np.isnan(xx[i]) or np.isnan(yy[i]):
                zones[i] = 255
                continue

            x = int(xx[i])
            y = int(yy[i])

            # sécurité : coordonnées hors image (regard perdu, clignement...)
            if 0 <= y < label.shape[0] and 0 <= x < label.shape[1]:
                zones[i] = label[y, x]
            else:
                # Amélioration : les coordonnées hors image sont également
                # considérées comme une perte de suivi.
                zones[i] = 255

        pbar.update(1)

    cap.release()
    pbar.close()

    # Ajout de la colonne contenant la zone associée à chaque point de regard.
    sync["zones"] = zones

    return sync

def classification_with_coor_bis2(video_path, coor):
    sync = coor
    coor = coor.to_numpy()
    n = len(coor)
    zones = [0] * n
    xx = coor[:, 0]
    yy = coor[:, 1]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Synchronisation")

    # Couleurs (ATTENTION OpenCV = BGR)
    colors = {
        1: np.array([97, 254, 0]),    # vert  -> Yeux
        2: np.array([99, 0, 252]),    # rose  -> Nez et Bouche
        3: np.array([94, 254, 253]),  # jaune -> Visage
        4: np.array([0, 0, 250]),     # rouge -> Haut du corps
        5: np.array([96, 0, 0]),      # bleu  -> Reste de la tête
    }
    TOLERANCE = 10  # marge d'erreur pour absorber les artefacts de compression

    temp = -3  # démarre à -3 pour que la 1re frame utilise range(0,3)

    while True:
        temp = temp + 3
        ret, frame = cap.read()
        if not ret:
            break

        # image label (par pixel), uint8 car les codes vont de 0 à 5
        label = np.zeros(frame.shape[:2], dtype=np.uint8)  # 0 = noir/autre

        # segmentation vectorisée avec tolérance sur la couleur
        for k, color in colors.items():
            diff = np.abs(frame.astype(np.int16) - color)
            mask = np.all(diff <= TOLERANCE, axis=-1)
            label[mask] = k
        # label contient maintenant la zone d'appartenance de chaque pixel

        for i in range(temp, temp + 3):
            if i >= n:
                break
            x, y = int(xx[i]), int(yy[i])
            # sécurité : coordonnées hors image (regard perdu, clignement...)
            if 0 <= y < label.shape[0] and 0 <= x < label.shape[1]:
                zones[i] = label[y, x]
            else:
                zones[i] = 0

        pbar.update(1)

    pbar.close()
    cap.release()

    # sync : contient les coordonnées avec leur zone correspondante dans la vidéo segmentée
    sync["zones"] = zones
    print("0 : Autre\n1 : Yeux\n2 : Nez et Bouche\n3 : Visage\n4 : Haut du corps\n5 : Reste de la tête\n")

    return sync


def classification_with_coor_bis(video_path, coor) :
    
    
    #video_path = sys.argv[1]
    sync = coor
    coor = coor.to_numpy()
    zones =len(coor)*[0]
    xx = coor[:,0]  #.to_numpy()
    yy = coor[:,1]  #.to_numpy()
    cap = cv2.VideoCapture(video_path)
    temp = -3
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Synchronisation")
    
    # Couleurs (ATTENTION OpenCV = BGR)
    colors = {
        1: np.array([97, 254, 0]),   # vert
        2: np.array([99, 0, 252]),   # rouge
        3: np.array([94, 254, 253]), # jaune
        4: np.array([0, 0, 250]),    # orange
        5: np.array([96, 0, 0])      # bleu
    }
    
    while True:
        temp = temp + 3
        ret, frame = cap.read()
        if not ret:
            break
    
        # image label (par pixel)
        label = np.full(frame.shape[:2], 0, dtype=np.uint8)  # 0 = noir/autre 
        # segmentation vectorisée
        for k, color in colors.items():
            mask = np.all(frame == color, axis=-1)
            label[mask] = k
        # label contient maintenant la zone d'appartenance de chaque pixel
        #print("taille label :",frame.shape[:2])
        for i in range(temp, temp+3 ):
            if i >= len(coor):
                break
            x , y = int(xx[i]) , int(yy[i])
            zones[i] = label[y,x]
        # zones contient le code de la couleur de la zone regardé
    
        pbar.update(1)
    
    pbar.close()
    cap.release()
    # sync : contient les coordonnées avec leurs zone correspondant dans la vidéo segmentée
    sync["zones"] = zones
    #zones = pd.DataFrame(zones, columns = ["Zones"])
    print("0 : Autre\n 1 : Yeux\n 2 : Nez et Bouche\n 3 : Visage\n 4 : Haut du corps\n 5 : Reste de la tête\n  ")
    os.system('espeak "Le traitement est terminé"')
    return sync



"""
print(label)
plt.imshow(label, cmap="jet")
plt.title("Segmentation des pixels")
plt.show()
"""


"""

video_path =  cv2.VideoCapture("/home/seydou/Bureau/Dossier_stage/données_stage/test/zones.wmv")

# Nombre total de frames de la video
print("Frames :", int(video_path.get(cv2.CAP_PROP_FRAME_COUNT)))

# Nombre de frames par seconde
fps = video_path.get(cv2.CAP_PROP_FPS)
print("FPS =", fps)

res = classification(video_path)
print(res)
"""
