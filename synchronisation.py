# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import time
import os
import classification_pixels as cp
#from traitement_video import *
import pandas as pd


"""
Ce programme parmet de faire la synchronisation entre la vidéo segmentée et fichier .txt issu du fichier .eta

La fonction centrale de ce programme est la fonction : classification_with_coor( arg1 , arg2)
En entrées :
_arg1 : est le chemin où se trouve la vidéo 
_arg2 : le chemin où se trouve les coordonnées du regard

Sorties : un data frame
"""


file = pd.read_csv("/home/seydou/Bureau/results_grid/PEP_sujet/PEP001.txt", sep = "\t")

video_path =  "/home/seydou/Bureau/results_grid/PEP_sujet/zones_sujet.wmv"

coor_eye = file[["Lft X Pos","Lft Y Pos"]]

# On calcul aussi la durée d'éxécution du programme

debut = time.time()

zones = cp.classification_with_coor(video_path, coor_eye)

fin = time.time()

print(f"Temps d'exécution : {(fin-debut)/60:.2f} minutes")

# Alarme pour prévenir quand l'exécution est terminé
os.system('notify-send "Segmentation terminée" "Le calcul est fini."')

os.system('espeak "Segmentation terminée"')

# Sauuvegarde de la video en fichier .csv
file["region"] = zones["zones"]
file.to_csv("/home/seydou/Bureau/results_grid/PEP_sujet/PEP001_after_sync.csv", index=False)
