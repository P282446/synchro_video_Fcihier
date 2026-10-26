import numpy as np
import cv2
from tqdm import tqdm

# =============================================================
# Couleurs de référence (définies en BGR)
# =============================================================
COLORS_BGR = {
    1: (0, 254, 97),      # vert  -> Yeux
    2: (252, 0, 99),      # rose  -> Nez et Bouche
    3: (253, 254, 94),    # jaune -> Visage
    4: (250, 0, 0),       # rouge -> Haut du corps
    5: (0, 0, 96),        # bleu  -> Reste de la tête
}


def classer_pixels(frame, colors, seuil):
    """
    Classe chaque pixel dans la couleur de référence la plus proche.
    Un pixel dont la distance euclidienne à toutes les références
    dépasse `seuil` reste à 0 (fond / non classé).
    Retourne (label, nb_pixels_colores).
    """
    keys = np.array(list(colors.keys()), dtype=np.uint8)
    refs = np.array(list(colors.values()), dtype=np.int32)       # (K, 3)

    f = frame.astype(np.int32)                                   # (H, W, 3)

    # Distance au carré à chaque couleur : (K, H, W)
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
    Compare le nombre de pixels reconnus si la vidéo est interprétée
    en BGR ou en RGB, sur quelques frames réparties dans la vidéo.
    Retourne "BGR" ou "RGB".
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = np.linspace(0, max(total - 1, 0), nb_echantillons).astype(int)

    colors_rgb = {k: tuple(v[::-1]) for k, v in colors_bgr.items()}
    score_bgr = 0
    score_rgb = 0

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue
        score_bgr += classer_pixels(frame, colors_bgr, seuil)[1]
        score_rgb += classer_pixels(frame, colors_rgb, seuil)[1]

    # On remet la vidéo au début
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Détection couleurs -> BGR : {score_bgr} px | RGB : {score_rgb} px")

    if score_bgr == 0 and score_rgb == 0:
        print("⚠ Aucune couleur reconnue : BGR utilisé par défaut.")

    return "BGR" if score_bgr >= score_rgb else "RGB"

import cv2
import numpy as np
from tqdm import tqdm

# Les fonctions suivantes restent celles de votre projet (inchangees) :
#   COLORS_BGR, detecter_ordre_couleurs(cap, COLORS_BGR, seuil_couleur),
#   classer_pixels(frame, colors, seuil_couleur) -> (label, total_colored_pixels)


def _col(df, nom):
    """Retrouve une colonne qu'elle s'appelle 'Lft X Pos' ou 'Lft.X.Pos'."""
    for c in (nom, nom.replace(" ", "."), nom.replace(" ", "_")):
        if c in df.columns:
            return c
    raise KeyError(f"Colonne introuvable : {nom}")


def classification_with_coor(video_path, coor,
                             color_order="auto",     # "auto", "BGR" ou "RGB"
                             seuil_couleur=40,
                             oeil="gauche",          # "gauche", "droit" ou "moyenne"
                             frame0=None,            # valeur de 'Frame' = 1re image de la video
                             decalage=0.0,           # secondes ajoutees au temps video
                             echelle_x=1.0,          # si la video n'a pas la taille de la scene
                             echelle_y=1.0):
    """
    Ajoute a `coor` :
      zones        : 1..5 = region, 254 = frame mal segmentee,
                     255 = regard perdu / hors image / hors video
      frame_video  : image de la video associee a la mesure
      t_video_s    : temps de la mesure sur l'horloge de la video (s)
      hors_video   : True si la mesure tombe avant/apres la video
    """

    # =========================================================
    # 1. Copie des donnees et coordonnees du regard
    # =========================================================
    sync = coor.copy()
    n = len(sync)
    if n == 0:
        sync["zones"] = np.zeros(0, dtype=np.uint8)
        return sync

    def xy(cote):
        px = sync[_col(sync, f"{cote} X Pos")].to_numpy(dtype=float).copy()
        py = sync[_col(sync, f"{cote} Y Pos")].to_numpy(dtype=float).copy()
        perdu = (px == 0) & (py == 0)          # (0, 0) = perte de suivi
        px[perdu], py[perdu] = np.nan, np.nan
        return px, py

    if oeil == "gauche":
        xx, yy = xy("Lft")
    elif oeil == "droit":
        xx, yy = xy("Rt")
    else:                                       # moyenne des yeux valides
        lx, ly = xy("Lft")
        rx, ry = xy("Rt")
        xx = np.nanmean(np.vstack([lx, rx]), axis=0)
        yy = np.nanmean(np.vstack([ly, ry]), axis=0)

    xx = xx * echelle_x
    yy = yy * echelle_y

    # =========================================================
    # 2. Temps de l'eye-tracker, ramene a l'horloge de la video
    # =========================================================
    # Capture Time est en microsecondes et ne commence PAS a zero :
    # on soustrait l'instant de reference (image 'frame0', sinon 1re mesure).
    ct = sync[_col(sync, "Capture Time")].to_numpy(dtype=np.float64)

    if frame0 is None:
        ref = 0
    else:
        lignes = np.flatnonzero(sync[_col(sync, "Frame")].to_numpy() == frame0)
        if len(lignes) == 0:
            raise ValueError(f"frame0={frame0} absent de la colonne Frame.")
        ref = int(lignes[0])

    temps_eye = (ct - ct[ref]) / 1_000_000 + decalage

    # =========================================================
    # 3. Ouverture de la video
    # =========================================================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video : {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Impossible de recuperer le FPS de la video.")

    print(f"Nombre de mesures eye-tracker : {n}")
    print(f"Nombre de frames video : {total}")
    print(f"FPS video : {fps:.3f}")
    print(f"Duree video : {total / fps:.3f} s")
    print(f"Duree eye-tracker : {(ct.max() - ct.min()) / 1e6:.3f} s "
          f"(reference : ligne {ref})")

    # =========================================================
    # 4. Synchronisation temporelle
    # =========================================================
    frame_eye = np.rint(temps_eye * fps).astype(np.int64)

    # Mesures avant/apres la video : on NE les ramene PAS sur la 1re/derniere
    # image (np.clip) : elles recevraient une region fausse. On les marque.
    hors_video = (frame_eye < 0) | (frame_eye >= total)
    print(f"Mesures hors video : {hors_video.sum()} ({hors_video.mean():.1%})")

    # =========================================================
    # 5. Couleurs des zones
    # =========================================================
    MIN_COLORED_PIXELS = 100
    nb_frames_invalides = 0

    if color_order == "auto":
        color_order = detecter_ordre_couleurs(cap, COLORS_BGR, seuil_couleur)

    if color_order == "RGB":
        colors = {k: tuple(v[::-1]) for k, v in COLORS_BGR.items()}
    else:
        colors = COLORS_BGR

    print(f"Ordre des couleurs utilise : {color_order}")
    print(f"Seuil de distance couleur : {seuil_couleur}")

    # =========================================================
    # 6. Organisation des mesures par frame video
    # =========================================================
    dans = np.flatnonzero(~hors_video)
    indices_tries = dans[np.argsort(frame_eye[dans], kind="stable")]
    frames_triees = frame_eye[indices_tries]

    debut = np.searchsorted(frames_triees, np.arange(total), side="left")
    fin = np.searchsorted(frames_triees, np.arange(total), side="right")

    # Tout est "perte / hors image" (255) par defaut
    zones = np.full(n, 255, dtype=np.uint8)

    # =========================================================
    # 7. Parcours de la video
    # =========================================================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # la detection BGR/RGB a deplace la lecture
    pbar = tqdm(total=total, desc="Synchronisation")

    try:
        for numero_frame in range(total):
            if not cap.grab():             # lit l'image sans la decoder
                break
            pbar.update(1)

            d, f = debut[numero_frame], fin[numero_frame]
            if d == f:                     # aucune mesure pour cette image
                continue

            ret, frame = cap.retrieve()    # decodage seulement si utile
            if not ret:
                continue

            indices = indices_tries[d:f]

            label, total_colored_pixels = classer_pixels(
                frame, colors, seuil_couleur)
            frame_invalide = total_colored_pixels < MIN_COLORED_PIXELS
            if frame_invalide:
                nb_frames_invalides += 1

            xs, ys = xx[indices], yy[indices]
            perdu = np.isnan(xs) | np.isnan(ys)
            res = np.full(len(indices), 255, dtype=np.uint8)

            if frame_invalide:
                res[~perdu] = 254
            else:
                xi = np.nan_to_num(xs, nan=-1).astype(np.int64)
                yi = np.nan_to_num(ys, nan=-1).astype(np.int64)
                h, w = label.shape[:2]
                dans_image = (~perdu) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                res[dans_image] = label[yi[dans_image], xi[dans_image]]

            zones[indices] = res
    finally:
        cap.release()
        pbar.close()

    # =========================================================
    # 8. Fin
    # =========================================================
    if nb_frames_invalides > 0:
        print(f"Attention : {nb_frames_invalides} frame(s) mal segmentee(s) "
              f"sur {total} frames au total.")

    sync["zones"] = zones
    sync["frame_video"] = frame_eye
    sync["t_video_s"] = temps_eye
    sync["hors_video"] = hors_video
    return sync

def classification_with_coor_V1(video_path, coor,
                             color_order="auto",    # "auto", "BGR" ou "RGB"
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

    # Vérification
    if n == 0:
        sync["zones"] = zones
        return sync

    # =========================================================
    # 2. Temps de l'eye-tracker
    # =========================================================

    # Capture Time est exprimé en microsecondes
    temps_eye = (
        sync["Capture Time"].to_numpy(dtype=np.float64)
        / 1_000_000
    )

    # =========================================================
    # 3. Ouverture de la vidéo
    # =========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Impossible d'ouvrir la vidéo : {video_path}"
        )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # Pour chaque mesure eye-tracker, on calcule la frame vidéo
    # dont le timestamp est le plus proche.
    #
    # Frame vidéo 0 -> temps 0
    # Frame vidéo 1 -> temps 1/fps
    # etc.

    frame_eye = np.rint(
        temps_eye * fps
    ).astype(np.int64)

    # Sécurité : empêcher d'obtenir une frame en dehors de la vidéo
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

    # Détection automatique de l'ordre des canaux (BGR ou RGB)
    if color_order == "auto":
        color_order = detecter_ordre_couleurs(
            cap, COLORS_BGR, seuil_couleur
        )

    if color_order == "RGB":
        colors = {k: tuple(v[::-1]) for k, v in COLORS_BGR.items()}
    else:
        colors = COLORS_BGR

    print(f"Ordre des couleurs utilisé : {color_order}")
    print(f"Seuil de distance couleur : {seuil_couleur}")

    # =========================================================
    # 6. Organisation des mesures par frame vidéo
    # =========================================================

    # indices_par_frame[0] = [0, 1]
    # indices_par_frame[1] = [2, 3, 4]
    # etc.
    # Le nombre de mesures peut être différent pour chaque frame.

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

    # La détection BGR/RGB a déplacé la lecture : on repart du début
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pbar = tqdm(
        total=total,
        desc="Synchronisation"
    )

    numero_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Sécurité si la vidéo contient plus de frames que annoncé
        if numero_frame >= total:
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

        indices = indices_tries[debut_frame:fin_frame]

        # -----------------------------------------------------
        # Classification des pixels par couleur la plus proche
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
