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


def classification_with_coor(video_path, coor) :
    
    
    #video_path = sys.argv[1]
    sync = coor
    coor = coor.to_numpy()
    zones =len(coor)*[0]
    xx = coor[:,0]  #.to_numpy()
    yy = coor[:,1]  #.to_numpy()
    cap = cv2.VideoCapture(video_path)
    temp = 0
    
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
        label = np.full(frame.shape[:2], 0, dtype=object)  # 0 = noir/autre 
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