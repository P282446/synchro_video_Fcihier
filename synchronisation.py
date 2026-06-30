# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""



import sys
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



if len(sys.argv) < 3:
    print("Usage : synchonisation.py video.vmw fichier.csv")
    sys.exit()


#file = pd.read_csv("/home/seydou/Bureau/results_grid/PEP_sujet/PEP001.txt", sep = "\t")
#video_path =  "/home/seydou/Bureau/results_grid/PEP_sujet/zones_sujet.wmv"

video_path = sys.argv[1]
file = sys.argv[2]

#On recupere le dossier ou se situe le file
output_dir = os.path.dirname(file)

# On cree le nom du fichier de sortie
base_name = os.path.splitext(os.path.basename(file))[0]
output_path = os.path.join(output_dir, base_name + "_result.csv")


f = pd.read_csv("file", sep = "\t")

# On recupere les coordonnees de fixations du regard
coor_eye = f[["Lft X Pos","Lft Y Pos"]]

# On calcul aussi la durée d'éxécution du programme

debut = time.time()

zones = cp.classification_with_coor(video_path, coor_eye)

fin = time.time()

print(f"Temps d'exécution : {(fin-debut)/60:.2f} minutes")

# Alarme pour prévenir quand l'exécution est terminé
os.system('notify-send "Segmentation terminée" "Le calcul est fini."')

os.system('espeak "Segmentation terminée"')

# Sauuvegarde de la video en fichier .csv
#f["region"] = zones["zones"]
#f.to_csv("/home/seydou/Bureau/results_grid/PEP_sujet/PEP001_after_sync.csv", index=False)
os.system('notify-send "Synchronisation terminée" "Le calcul est fini."')

# Sauuvegarde de la video en fichier .csv
f["region"] = zones["zones"]
f.to_csv(output_path, index=False)


os.system('espeak "Synchronisation terminée"')

