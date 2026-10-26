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




# Couleurs de reference definies en BGR
COLORS_BGR = {
    1: (97, 254, 0),      # vert  -> Yeux
    2: (99, 0, 252),      # rose  -> Nez et Bouche
    3: (94, 254, 253),    # jaune -> Visage
    4: (0, 0, 250),       # rouge -> Haut du corps
    5: (96, 0, 0),        # bleu  -> Reste de la tete
    }


def classer_pixels(frame, colors, seuil):
    """
    Classe chaque pixel dans la couleur de reference la plus proche.
    Un pixel dont la distance (euclidienne) a toutes les references
    depasse `seuil` reste a 0 (fond / non classe).
    Retourne (label, nb_pixels_colores).
    """
    keys = np.array(list(colors.keys()), dtype=np.uint8)
    refs = np.array(list(colors.values()), dtype=np.int32)       # (K, 3)

    f = frame.astype(np.int32)                                   # (H, W, 3)

    # Distance au carre a chaque couleur : (K, H, W)
    dist2 = np.stack(
        [np.sum((f - ref) ** 2, axis=-1) for ref in refs]
    )

    idx = np.argmin(dist2, axis=0)                               # (H, W)
    dmin = np.take_along_axis(dist2, idx[None], axis=0)[0]

    masque = dmin <= seuil ** 2
    label = np.where(masque, keys[idx], 0).astype(np.uint8)

    return label, int(np.count_nonzero(masque))


def detecter_ordre_couleurs(cap, colors_bgr, seuil, nb_echantillons=20):
    """
    Compare le nombre de pixels reconnus si la video est interpretee
    en BGR ou en RGB, sur quelques frames reparties dans la video.
    Retourne "BGR" ou "RGB".
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = np.linspace(0, max(total - 1, 0), nb_echantillons).astype(int)

    colors_rgb = {k: v[::-1] for k, v in colors_bgr.items()}
    score_bgr = 0
    score_rgb = 0

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue
        score_bgr += classer_pixels(frame, colors_bgr, seuil)[1]
        score_rgb += classer_pixels(frame, colors_rgb, seuil)[1]

    # On remet la video au debut
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Detection couleurs -> BGR : {score_bgr} px | RGB : {score_rgb} px")
    return "BGR" if score_bgr >= score_rgb else "RGB"


def classification_with_coor(video_path, coor,
                             color_order="auto",   # "auto", "BGR" ou "RGB"
                             seuil_couleur=40):

    # =========================================================
    # 1. Copie des données
    # =========================================================

    sync = coor.copy()

    # Les coordonnées (0,0) correspondent à une perte de suivi
    mask = (
        (sync["Lft X Pos"] == 0) &
        (sync["Lft Y Pos"] == 0)
    )

    sync.loc[
        mask,
        ["Lft X Pos", "Lft Y Pos"]
    ] = np.nan

    # Coordonnées
    xx = sync["Lft X Pos"].to_numpy()
    yy = sync["Lft Y Pos"].to_numpy()

    n = len(sync)

    # Tableau contenant la zone de chaque mesure eye-tracker
    zones = np.zeros(n, dtype=np.uint8)

    # =========================================================
    # 2. Temps de l'eye-tracker
    # =========================================================

    # Capture.Time est exprimé en microsecondes
    temps_eye = (
        sync["Capture Time"].to_numpy(dtype=np.float64)
        / 1_000_000
    )

    # Vérification
    if n == 0:
        sync["zones"] = zones
        return sync

    # =========================================================
    # 3. Ouverture de la vidéo
    # =========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Impossible d'ouvrir la vidéo : {video_path}"
        )

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError(
            "Impossible de récupérer le FPS de la vidéo."
        )

    duree_video = total / fps
    duree_eye = temps_eye[-1]

    print(f"Nombre de mesures eye-tracker : {n}")
    print(f"Nombre de frames vidéo : {total}")
    print(f"FPS vidéo : {fps:.3f}")
    print(f"Durée vidéo : {duree_video:.3f} s")
    print(f"Durée eye-tracker : {duree_eye:.3f} s")

    # =========================================================
    # 4. Synchronisation temporelle
    # =========================================================

    # Pour chaque mesure eye-tracker, on calcule directement
    # la frame vidéo dont le timestamp est le plus proche.
    #
    # Frame vidéo 0 -> temps 0
    # Frame vidéo 1 -> temps 1/fps
    # Frame vidéo 2 -> temps 2/fps
    # etc.

    frame_eye = np.rint(
        temps_eye * fps
    ).astype(np.int64)

    # Sécurité : empêcher d'obtenir une frame en dehors
    # de la vidéo.
    frame_eye = np.clip(
        frame_eye,
        0,
        total - 1
    )

    # =========================================================
    # 5. Couleurs des zones
    # =========================================================

    MIN_COLORED_PIXELS = 100
    nb_frames_invalides = 0
    label, total_colored_pixels = classer_pixels(
            frame, colors, seuil_couleur
        )
    if color_order == "auto":
        color_order = detecter_ordre_couleurs(cap, COLORS_BGR, seuil_couleur)

    if color_order == "RGB":
        colors = {k: v[::-1] for k, v in COLORS_BGR.items()}
    else:
        colors = COLORS_BGR

    print(f"Ordre des couleurs utilise : {color_order}")

    # =========================================================
    # 6. Organisation des mesures par frame vidéo
    # =========================================================

    # On construit une liste contenant les indices des mesures
    # eye-tracker correspondant à chaque frame vidéo.
    #
    # Exemple :
    #
    # indices_par_frame[0] = [0, 1]
    # indices_par_frame[1] = [2, 3, 4]
    # indices_par_frame[2] = [5, 6]
    #
    # Le nombre de mesures peut donc être différent pour
    # chaque frame.

    indices_tries = np.argsort(frame_eye)

    frames_triees = frame_eye[indices_tries]

    debut = np.searchsorted(
        frames_triees,
        np.arange(total),
        side="left"
    )

    fin = np.searchsorted(
        frames_triees,
        np.arange(total),
        side="right"
    )

    # =========================================================
    # 7. Parcours de la vidéo
    # =========================================================

    pbar = tqdm(
        total=total,
        desc="Synchronisation"
    )

    numero_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------------------
        # Mesures eye-tracker associées à cette frame
        # -----------------------------------------------------

        debut_frame = debut[numero_frame]
        fin_frame = fin[numero_frame]

        if debut_frame == fin_frame:
            # Aucune mesure eye-tracker pour cette frame
            pbar.update(1)
            numero_frame += 1
            continue

        indices = indices_tries[
            debut_frame:fin_frame
        ]

        # -----------------------------------------------------
        # Création du masque des couleurs
        # -----------------------------------------------------

        label, total_colored_pixels = classer_pixels(
            frame, colors, seuil_couleur
        )

        # -----------------------------------------------------
        # Vérification de la segmentation
        # -----------------------------------------------------

        frame_invalide = (
            total_colored_pixels
            < MIN_COLORED_PIXELS
        )

        if frame_invalide:
            nb_frames_invalides += 1

        # -----------------------------------------------------
        # Classification des mesures eye-tracker
        # -----------------------------------------------------

        for i in indices:

            # Perte de suivi
            if (
                np.isnan(xx[i])
                or np.isnan(yy[i])
            ):

                zones[i] = 255
                continue

            # Frame vidéo mal segmentée
            if frame_invalide:

                zones[i] = 254
                continue

            x = int(xx[i])
            y = int(yy[i])

            # Coordonnées dans l'image
            if (
                0 <= y < label.shape[0]
                and
                0 <= x < label.shape[1]
            ):

                zones[i] = label[y, x]

            else:

                # Coordonnées hors image
                zones[i] = 255

        pbar.update(1)
        numero_frame += 1

    # =========================================================
    # 8. Fin
    # =========================================================

    cap.release()
    pbar.close()

    if nb_frames_invalides > 0:

        print(
            f"⚠ {nb_frames_invalides} frame(s) mal "
            f"segmentée(s) détectée(s) et ignorée(s) "
            f"sur {total} frames au total."
        )

    # Ajout des zones
    sync["zones"] = zones

    return sync



def classification_with_coor6(video_path, coor):

    # Copie du DataFrame afin de ne pas modifier les données d'origine
    sync = coor.copy()

    # ---------------------------------------------------------
    # 1. Préparation des coordonnées de l'eye-tracker
    # ---------------------------------------------------------

    # Les coordonnées (0,0) correspondent à une perte de suivi
    mask = (
        (sync["Lft X Pos"] == 0) &
        (sync["Lft Y Pos"] == 0)
    )

    sync.loc[
        mask,
        ["Lft X Pos", "Lft Y Pos"]
    ] = np.nan

    # ---------------------------------------------------------
    # 2. Temps de l'eye-tracker
    # ---------------------------------------------------------

    # Capture.Time est en microsecondes
    # Conversion en secondes
    temps_eye = sync["Capture Time"].to_numpy() / 1_000_000

    coor = sync.to_numpy()

    n = len(coor)

    zones = np.zeros(n, dtype=np.uint8)

    xx = sync["Lft X Pos"].to_numpy()
    yy = sync["Lft Y Pos"].to_numpy()

    # ---------------------------------------------------------
    # 3. Ouverture de la vidéo
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise ValueError(
            "Impossible de récupérer le FPS de la vidéo."
        )

    print(f"Nombre de frames vidéo : {total}")
    print(f"FPS vidéo : {fps}")
    print(f"Durée vidéo : {total / fps:.3f} s")

    print(
        f"Durée eye-tracker : "
        f"{temps_eye[-1]:.3f} s"
    )

    pbar = tqdm(
        total=total,
        desc="Synchronisation"
    )

    # ---------------------------------------------------------
    # 4. Couleurs des zones
    # ---------------------------------------------------------

    # ATTENTION : OpenCV utilise BGR
    colors = {
        1: np.array([97, 254, 0]),     # vert  -> Yeux
        2: np.array([99, 0, 252]),     # rose  -> Nez et Bouche
        3: np.array([94, 254, 253]),  # jaune -> Visage
        4: np.array([0, 0, 250]),     # rouge -> Haut du corps
        5: np.array([96, 0, 0]),      # bleu  -> Reste de la tête
    }

    TOLERANCE = 10

    MIN_COLORED_PIXELS = 100

    nb_frames_invalides = 0

    # ---------------------------------------------------------
    # 5. Synchronisation temporelle
    # ---------------------------------------------------------

    # Indice de la prochaine mesure eye-tracker à examiner
    indice_eye = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Numéro de la frame vidéo
        numero_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Temps de cette frame dans la vidéo
        temps_video = numero_frame / fps

        # -----------------------------------------------------
        # Recherche des mesures eye-tracker correspondant
        # au temps de cette frame vidéo
        # -----------------------------------------------------

        indices_frame = []

        # On avance dans les mesures eye-tracker jusqu'à
        # atteindre le temps de la frame vidéo
        while (
            indice_eye < n
            and temps_eye[indice_eye] <= temps_video + (1 / (2 * fps))
        ):

            # Cette mesure appartient à la frame vidéo
            indices_frame.append(indice_eye)

            indice_eye += 1

        # -----------------------------------------------------
        # Si aucune mesure eye-tracker ne correspond
        # -----------------------------------------------------

        if len(indices_frame) == 0:

            pbar.update(1)

            continue

        # -----------------------------------------------------
        # Image label (par pixel)
        # -----------------------------------------------------

        label = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        # Conversion en int16 pour éviter les dépassements
        frame16 = frame.astype(np.int16)

        # -----------------------------------------------------
        # Segmentation vectorisée
        # -----------------------------------------------------

        total_colored_pixels = 0

        for k, color in colors.items():

            diff = np.abs(frame16 - color)

            color_mask = np.all(
                diff <= TOLERANCE,
                axis=-1
            )

            label[color_mask] = k

            total_colored_pixels += np.count_nonzero(
                color_mask
            )

        # -----------------------------------------------------
        # Vérification de la frame
        # -----------------------------------------------------

        frame_invalide = (
            total_colored_pixels <
            MIN_COLORED_PIXELS
        )

        if frame_invalide:
            nb_frames_invalides += 1

        # -----------------------------------------------------
        # Classification des mesures eye-tracker
        # correspondant à cette frame vidéo
        # -----------------------------------------------------

        for i in indices_frame:

            # Perte de suivi
            if np.isnan(xx[i]) or np.isnan(yy[i]):

                zones[i] = 255

                continue

            # Frame vidéo mal segmentée
            if frame_invalide:

                zones[i] = 254

                continue

            x = int(xx[i])
            y = int(yy[i])

            # Vérification que le point est dans l'image
            if (
                0 <= y < label.shape[0]
                and
                0 <= x < label.shape[1]
            ):

                zones[i] = label[y, x]

            else:

                # Coordonnées hors image
                zones[i] = 255

        pbar.update(1)

    # ---------------------------------------------------------
    # 6. Fin
    # ---------------------------------------------------------

    cap.release()

    pbar.close()

    if nb_frames_invalides > 0:

        print(
            f"⚠ {nb_frames_invalides} frame(s) mal "
            f"segmentée(s) détectée(s) et ignorée(s) "
            f"sur {total} frames au total."
        )

    # Ajout des zones
    sync["zones"] = zones

    return sync

def classification_with_coor5(video_path, coor):

    # =========================================================
    # 1. Copie des données
    # =========================================================

    sync = coor.copy()

    # Les coordonnées (0,0) correspondent à une perte de suivi
    mask = (
        (sync["Lft X Pos"] == 0) &
        (sync["Lft Y Pos"] == 0)
    )

    sync.loc[
        mask,
        ["Lft X Pos", "Lft Y Pos"]
    ] = np.nan

    # Coordonnées
    xx = sync["Lft X Pos"].to_numpy()
    yy = sync["Lft Y Pos"].to_numpy()

    n = len(sync)

    # Tableau contenant la zone de chaque mesure eye-tracker
    zones = np.zeros(n, dtype=np.uint8)

    # =========================================================
    # 2. Temps de l'eye-tracker
    # =========================================================

    # Capture.Time est exprimé en microsecondes
    temps_eye = (
        sync["Capture Time"].to_numpy(dtype=np.float64)
        / 1_000_000
    )

    # Vérification
    if n == 0:
        sync["zones"] = zones
        return sync

    # =========================================================
    # 3. Ouverture de la vidéo
    # =========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Impossible d'ouvrir la vidéo : {video_path}"
        )

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError(
            "Impossible de récupérer le FPS de la vidéo."
        )

    duree_video = total / fps
    duree_eye = temps_eye[-1]

    print(f"Nombre de mesures eye-tracker : {n}")
    print(f"Nombre de frames vidéo : {total}")
    print(f"FPS vidéo : {fps:.3f}")
    print(f"Durée vidéo : {duree_video:.3f} s")
    print(f"Durée eye-tracker : {duree_eye:.3f} s")

    # =========================================================
    # 4. Synchronisation temporelle
    # =========================================================

    # Pour chaque mesure eye-tracker, on calcule directement
    # la frame vidéo dont le timestamp est le plus proche.
    #
    # Frame vidéo 0 -> temps 0
    # Frame vidéo 1 -> temps 1/fps
    # Frame vidéo 2 -> temps 2/fps
    # etc.

    frame_eye = np.rint(
        temps_eye * fps
    ).astype(np.int64)

    # Sécurité : empêcher d'obtenir une frame en dehors
    # de la vidéo.
    frame_eye = np.clip(
        frame_eye,
        0,
        total - 1
    )

    # =========================================================
    # 5. Couleurs des zones
    # =========================================================

    # ATTENTION : OpenCV utilise BGR
    colors = {
        1: np.array([97, 254, 0]),      # vert  -> Yeux
        2: np.array([99, 0, 252]),      # rose  -> Nez et Bouche
        3: np.array([94, 254, 253]),    # jaune -> Visage
        4: np.array([0, 0, 250]),       # rouge -> Haut du corps
        5: np.array([96, 0, 0]),        # bleu  -> Reste de la tête
    }

    TOLERANCE = 10

    MIN_COLORED_PIXELS = 100

    nb_frames_invalides = 0

    # =========================================================
    # 6. Organisation des mesures par frame vidéo
    # =========================================================

    # On construit une liste contenant les indices des mesures
    # eye-tracker correspondant à chaque frame vidéo.
    #
    # Exemple :
    #
    # indices_par_frame[0] = [0, 1]
    # indices_par_frame[1] = [2, 3, 4]
    # indices_par_frame[2] = [5, 6]
    #
    # Le nombre de mesures peut donc être différent pour
    # chaque frame.

    indices_tries = np.argsort(frame_eye)

    frames_triees = frame_eye[indices_tries]

    debut = np.searchsorted(
        frames_triees,
        np.arange(total),
        side="left"
    )

    fin = np.searchsorted(
        frames_triees,
        np.arange(total),
        side="right"
    )

    # =========================================================
    # 7. Parcours de la vidéo
    # =========================================================

    pbar = tqdm(
        total=total,
        desc="Synchronisation"
    )

    numero_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------------------
        # Mesures eye-tracker associées à cette frame
        # -----------------------------------------------------

        debut_frame = debut[numero_frame]
        fin_frame = fin[numero_frame]

        if debut_frame == fin_frame:
            # Aucune mesure eye-tracker pour cette frame
            pbar.update(1)
            numero_frame += 1
            continue

        indices = indices_tries[
            debut_frame:fin_frame
        ]

        # -----------------------------------------------------
        # Création du masque des couleurs
        # -----------------------------------------------------

        label = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        frame16 = frame.astype(np.int16)

        total_colored_pixels = 0

        for k, color in colors.items():

            diff = np.abs(
                frame16 - color
            )

            color_mask = np.all(
                diff <= TOLERANCE,
                axis=-1
            )

            label[color_mask] = k

            total_colored_pixels += np.count_nonzero(
                color_mask
            )

        # -----------------------------------------------------
        # Vérification de la segmentation
        # -----------------------------------------------------

        frame_invalide = (
            total_colored_pixels
            < MIN_COLORED_PIXELS
        )

        if frame_invalide:
            nb_frames_invalides += 1

        # -----------------------------------------------------
        # Classification des mesures eye-tracker
        # -----------------------------------------------------

        for i in indices:

            # Perte de suivi
            if (
                np.isnan(xx[i])
                or np.isnan(yy[i])
            ):

                zones[i] = 255
                continue

            # Frame vidéo mal segmentée
            if frame_invalide:

                zones[i] = 254
                continue

            x = int(xx[i])
            y = int(yy[i])

            # Coordonnées dans l'image
            if (
                0 <= y < label.shape[0]
                and
                0 <= x < label.shape[1]
            ):

                zones[i] = label[y, x]

            else:

                # Coordonnées hors image
                zones[i] = 255

        pbar.update(1)
        numero_frame += 1

    # =========================================================
    # 8. Fin
    # =========================================================

    cap.release()
    pbar.close()

    if nb_frames_invalides > 0:

        print(
            f"⚠ {nb_frames_invalides} frame(s) mal "
            f"segmentée(s) détectée(s) et ignorée(s) "
            f"sur {total} frames au total."
        )

    # Ajout des zones
    sync["zones"] = zones

    return sync


def classification_with_coor4(video_path, coor):

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
