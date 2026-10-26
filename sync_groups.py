#!/usr/bin/env python3
"""
Synchronise automatiquement le regard (eye-tracker) avec les videos segmentees
de TOUS les participants, puis cree un seul fichier : resultats_analyse.csv

  Entrees :  ~/Bureau/fichiers/NOM.txt     regard
             ~/Bureau/videos/seg_NOM.*     video segmentee (meme NOM)
  Sorties :  ~/Bureau/resultats_synchro/NOM.csv           un fichier par video
             ~/Bureau/resultats_synchro/controle_synchro.csv   verifications
             ~/Bureau/resultats_analyse.csv                tout fusionne

Reprenable : les videos deja traitees sont sautees si vous relancez.
Lancement  : python synchroniser_tout.py
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

try:                       # barre de progression (pip install tqdm)
    from tqdm import tqdm
except ImportError:
    tqdm = None

# =============================================================================
# A REMPLIR : couleurs de la segmentation
# Format : {code_de_la_region: (B, G, R)}   (ordre OpenCV : bleu, vert, rouge)
# Valeurs de depart : palette de votre script R pour Excel (couleurs pastel),
# convertie en B, G, R. Le programme VERIFIE ces couleurs sur la vraie video avant
# de commencer ; si elles ne correspondent pas, il affiche les couleurs reelles :
# copiez-les ici (ou collez votre dictionnaire COLORS_BGR).
# =============================================================================
PALETTE_BGR = {
    1: (0, 254, 97),      # vert  -> Yeux
    2: (252, 0, 99),      # rose  -> Nez et Bouche
    3: (253, 254, 94),    # jaune -> Visage
    4: (250, 0, 0),       # rouge -> Haut du corps
    5: (0, 0, 96),        # bleu  -> Reste de la tête
}


TOLERANCE = 40.0           # distance de couleur maximale (compression video)
PERTE, MAL_SEGMENTE = 255, 254


# -----------------------------------------------------------------------------
# Lecture du regard
# -----------------------------------------------------------------------------
def lire_regard(chemin):
    df = pd.read_csv(chemin, sep=None, engine="python")
    df.columns = [str(c).strip().replace(" ", ".") for c in df.columns]
    requis = ["Obsv", "Frame", "Capture.Time", "Lft.X.Pos", "Lft.Y.Pos",
              "Rt.X.Pos", "Rt.Y.Pos"]
    manque = [c for c in requis if c not in df.columns]
    if manque:
        raise ValueError(f"colonnes absentes {manque} "
                         f"(trouvees : {list(df.columns)})")
    return df


def coordonnees(df, oeil):
    lx, ly = df["Lft.X.Pos"].to_numpy(float), df["Lft.Y.Pos"].to_numpy(float)
    rx, ry = df["Rt.X.Pos"].to_numpy(float), df["Rt.Y.Pos"].to_numpy(float)
    lv, rv = (lx > 0) | (ly > 0), (rx > 0) | (ry > 0)     # (0, 0) = oeil perdu
    if oeil == "gauche":
        return lx, ly, lv
    if oeil == "droit":
        return rx, ry, rv
    x = np.where(lv & rv, (lx + rx) / 2, np.where(lv, lx, rx))
    y = np.where(lv & rv, (ly + ry) / 2, np.where(lv, ly, ry))
    return x, y, lv | rv


# -----------------------------------------------------------------------------
# Lecture de la video segmentee
# -----------------------------------------------------------------------------
def ordre_couleurs(cap, pal):
    """Compare (B,G,R) et (R,G,B) sur quelques images et garde le meilleur."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores = {"BGR": 0.0, "RGB": 0.0}
    for i in np.linspace(0, max(n - 1, 0), 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        pix = f[::4, ::4].reshape(-1, 3).astype(np.int32)
        for nom, p in (("BGR", pal), ("RGB", pal[:, ::-1])):
            d = np.sqrt(((pix[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
            scores[nom] += float((d <= TOLERANCE).mean())
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return "RGB" if scores["RGB"] > scores["BGR"] else "BGR"


def verifier_palette(video, palette):
    """Compare la palette a de vraies images de la video AVANT le long calcul."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    morceaux = []
    for i in np.linspace(0, max(n - 1, 0), 6).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            morceaux.append(f[::3, ::3].reshape(-1, 3))
    cap.release()
    if not morceaux:
        sys.exit(f"Video illisible : {video}")
    pix = np.vstack(morceaux).astype(np.int32)
    codes = np.array(sorted(palette))
    pal = np.array([palette[c] for c in codes], dtype=np.int32)

    def parts(p):      # % de pixels proches de chaque couleur de la palette
        d = np.sqrt(((pix[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        return (d <= TOLERANCE).mean(0) * 100

    bgr, rgb = parts(pal), parts(pal[:, ::-1])
    ordre, pc = ("RGB", rgb) if rgb.sum() > bgr.sum() else ("BGR", bgr)
    print(f"Verification de la palette sur {os.path.basename(video)} (canaux {ordre}) :")
    for c, v in zip(codes, pc):
        print(f"  code {c} : {v:5.1f}% des pixels")

    if pc.max() < 0.5:
        q = (pix // 8) * 8 + 4
        vals, cnt = np.unique(q, axis=0, return_counts=True)
        top = np.argsort(-cnt)[:8]
        lignes = "\n".join(f"  (B, G, R) = {tuple(int(v) for v in vals[i])} : "
                           f"{cnt[i] / len(pix) * 100:.1f}% des pixels" for i in top)
        sys.exit("\nAucune couleur de PALETTE_BGR ne se retrouve dans cette video.\n"
                 "Couleurs les plus frequentes dans la video (a copier dans "
                 f"PALETTE_BGR avec le bon code) :\n{lignes}\n")
    absents = [int(c) for c, v in zip(codes, pc) if v < 0.05]
    if absents:
        print(f"  Remarque : aucun pixel des codes {absents} dans ces images "
              "(normal si ces regions n'y apparaissent pas).")


def decoder(frame, x, y, pal, codes, demi=2):
    """Code de la region sous le point (x, y) : vote sur un carre de 5 x 5 pixels."""
    h, w = frame.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return PERTE
    p = frame[max(0, yi - demi):yi + demi + 1,
              max(0, xi - demi):xi + demi + 1].reshape(-1, 3).astype(np.int32)
    d = ((p[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
    k, dm = d.argmin(1), np.sqrt(d.min(1))
    ok = dm <= TOLERANCE
    if not ok.any():
        return MAL_SEGMENTE
    return int(codes[np.bincount(k[ok], minlength=len(codes)).argmax()])


# -----------------------------------------------------------------------------
# Traitement d'UNE video (execute dans un processus de calcul)
# -----------------------------------------------------------------------------
def traiter_video(nom, regard, video, dossier, frame0, decalage, oeil, palette):
    cv2.setNumThreads(1)
    debut = time.time()

    df = lire_regard(regard)
    x, y, valide = coordonnees(df, oeil)
    ct = df["Capture.Time"].to_numpy(float)

    if frame0 is None:
        ref = 0
    else:
        lignes = np.flatnonzero(df["Frame"].to_numpy() == frame0)
        if len(lignes) == 0:
            raise ValueError(f"frame0={frame0} absent de la colonne Frame")
        ref = int(lignes[0])
    t_s = (ct - ct[ref]) / 1e6 + decalage

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"video illisible : {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or n_frames <= 0:
        raise ValueError("cadence ou nombre d'images de la video introuvable")

    codes = np.array(sorted(palette))
    pal = np.array([palette[c] for c in codes], dtype=np.int32)
    ordre = ordre_couleurs(cap, pal)
    if ordre == "RGB":
        pal = pal[:, ::-1]

    vf = np.rint(t_s * fps).astype(np.int64)         # image video de chaque mesure
    dans = (vf >= 0) & (vf < n_frames)
    region = np.full(len(df), PERTE, dtype=np.uint8)

    sel = np.flatnonzero(dans & valide)
    tri = sel[np.argsort(vf[sel], kind="stable")]
    cles, debuts = np.unique(vf[tri], return_index=True)
    fins = np.append(debuts[1:], len(tri))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    for cle, d, f in zip(cles, debuts, fins):
        while idx < cle:                              # saute les images inutiles
            cap.grab()
            idx += 1
        ok, frame = cap.read()
        idx += 1
        if not ok:
            continue
        for i in tri[d:f]:
            region[i] = decoder(frame, x[i], y[i], pal, codes)
    cap.release()

    out = pd.DataFrame({
        "Obsv": df["Obsv"], "Frame": df["Frame"],
        "Capture.Time": np.rint(t_s * 1e6).astype("int64"),
        "t_video_s": t_s.round(4), "video_frame": vf,
        "x": np.round(x, 1), "y": np.round(y, 1),
        "region": region, "saccade": np.nan,
    })
    chemin = os.path.join(dossier, f"{nom}.csv")
    out.to_csv(chemin + ".tmp", index=False)
    os.replace(chemin + ".tmp", chemin)       # jamais de fichier a moitie ecrit

    duree_video = n_frames / fps
    duree_regard = (ct.max() - ct.min()) / 1e6
    reg_dans = region[dans]
    pct_hors = float((~dans).mean() * 100)
    pct_perte = float((reg_dans == PERTE).mean() * 100) if len(reg_dans) else 0.0
    pct_254 = float((reg_dans == MAL_SEGMENTE).mean() * 100) if len(reg_dans) else 0.0

    # part de la video qui a au moins une mesure de regard
    couverture = len(np.unique(vf[dans])) / n_frames * 100

    alertes = []
    if frame0 is None:
        # sans frame0, on suppose que regard et video commencent ensemble
        if abs(duree_regard - duree_video) > 5:
            alertes.append(f"regard {duree_regard - duree_video:+.0f} s par rapport a la "
                           f"video : verifier frame0")
        if pct_hors > 5:
            alertes.append(f"{pct_hors:.0f}% des mesures hors video : verifier frame0")
    if couverture < 95:
        alertes.append(f"le regard ne couvre que {couverture:.0f}% de la video")
    if pct_254 > 20:
        alertes.append(f"{pct_254:.0f}% hors palette : verifier PALETTE_BGR")

    stats = {
        "video": nom, "duree_video_s": round(duree_video, 1),
        "duree_regard_s": round(duree_regard, 1),
        "ecart_s": round(duree_regard - duree_video, 1),
        "pct_hors_video": round(pct_hors, 1),
        "pct_video_couverte": round(couverture, 1),
        "pct_regard_perdu_255": round(pct_perte, 1),
        "pct_hors_palette_254": round(pct_254, 1),
        "ordre_couleurs": ordre,
        "frame0": "-" if frame0 is None else frame0, "decalage_s": decalage,
        "temps_calcul_s": round(time.time() - debut),
        "alerte": "; ".join(alertes),
    }
    with open(os.path.join(dossier, f"{nom}.stats.json"), "w", encoding="utf-8") as fj:
        json.dump(stats, fj)
    return stats


# -----------------------------------------------------------------------------
# Programme principal
# -----------------------------------------------------------------------------
def decouper_nom(nom):
    """AMP-084-Psy-Entr2 -> ('AMP-084', 'psy') ; REY-072-sUJET -> ('REY-072', 'sujet')."""
    participant = "-".join(nom.split("-")[:2])
    bas = nom.lower()
    return participant, ("psy" if "psy" in bas else "sujet" if "sujet" in bas else "")


def lire_decalages(chemin):
    """Fichier optionnel (CSV : nom,frame0,decalage) : un reglage par video."""
    res = {}
    if not chemin:
        return res
    d = pd.read_csv(chemin, dtype=str).fillna("")
    for _, r in d.iterrows():
        f0 = r.get("frame0", "").strip()
        res[r["nom"].strip()] = (None if f0 in ("", "-") else int(float(f0)),
                                 float(r.get("decalage", "") or 0))
    return res


def main():
    home = os.path.expanduser("~/Bureau")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fichiers", default=f"{home}/fichiers")
    ap.add_argument("--videos", default=f"{home}/videos")
    ap.add_argument("--dossier-resultats", default=f"{home}/resultats_synchro")
    ap.add_argument("--sortie", default=f"{home}/resultats_analyse.csv")
    ap.add_argument("--workers", type=int, default=0,
                    help="videos traitees en meme temps (defaut : tous les coeurs reserves)")
    ap.add_argument("--oeil", choices=["moyenne", "gauche", "droit"], default="moyenne")
    ap.add_argument("--decalages", default=None,
                    help="CSV optionnel (nom,frame0,decalage) pour regler une video")
    ap.add_argument("--sans-verif-palette", action="store_true",
                    help="ne verifie pas les couleurs sur la video avant de commencer")
    ap.add_argument("--refaire", action="store_true",
                    help="retraite aussi les videos deja faites")
    a = ap.parse_args()

    if not PALETTE_BGR:
        sys.exit("Il faut d'abord remplir PALETTE_BGR en haut de ce fichier "
                 "(couleurs de la segmentation).")

    os.makedirs(a.dossier_resultats, exist_ok=True)
    reglages = lire_decalages(a.decalages)

    # ---- appariement regard <-> video : meme NOM, la video commence par seg_
    paires, sans_video = [], []
    for r in sorted(glob.glob(os.path.join(a.fichiers, "*.txt"))):
        nom = os.path.splitext(os.path.basename(r))[0]
        vids = sorted(glob.glob(os.path.join(a.videos, f"seg_{glob.escape(nom)}.*")))
        (paires if vids else sans_video).append((nom, r, vids[0]) if vids else nom)
    print(f"{len(paires)} paires regard/video trouvees.")
    for nom in sans_video:
        print(f"  ATTENTION : pas de video seg_{nom}.* -> ignore")
    if not paires:
        sys.exit("Aucune paire trouvee : verifiez --fichiers et --videos.")

    def fait(nom):
        base = os.path.join(a.dossier_resultats, nom)
        return os.path.getsize(base + ".csv") > 0 if os.path.exists(base + ".csv") \
            and os.path.exists(base + ".stats.json") else False

    a_faire = [p for p in paires if a.refaire or not fait(p[0])]
    print(f"{len(paires) - len(a_faire)} deja traitees, {len(a_faire)} a traiter.")

    def ecrire(msg):
        (tqdm.write if tqdm else lambda m: print(m, flush=True))(msg)

    if a_faire and not a.sans_verif_palette:
        verifier_palette(a_faire[0][2], PALETTE_BGR)

    # ---- traitement en parallele
    echecs = []
    if a_faire:
        try:
            coeurs = len(os.sched_getaffinity(0))
        except AttributeError:
            coeurs = os.cpu_count() or 1
        workers = max(1, min(a.workers or coeurs, len(a_faire)))
        print(f"Traitement avec {workers} processus en parallele.")
        barre = tqdm(total=len(a_faire), desc="Videos", unit="video") if tqdm else None
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for nom, r, v in a_faire:
                f0, dec = reglages.get(nom, (None, 0.0))
                futs[ex.submit(traiter_video, nom, r, v, a.dossier_resultats,
                               f0, dec, a.oeil, PALETTE_BGR)] = nom
            for k, fut in enumerate(as_completed(futs), 1):
                nom = futs[fut]
                try:
                    st = fut.result()
                    note = f" | ATTENTION : {st['alerte']}" if st["alerte"] else ""
                    ecrire(f"[{k}/{len(a_faire)}] {nom} termine en "
                           f"{st['temps_calcul_s'] / 60:.1f} min | "
                           f"regard perdu {st['pct_regard_perdu_255']}%{note}")
                except Exception as e:  # noqa: BLE001
                    echecs.append(nom)
                    ecrire(f"[{k}/{len(a_faire)}] {nom} ECHEC : {e}")
                if barre:
                    barre.update(1)
        if barre:
            barre.close()

    # ---- fichier de controle
    stats = []
    for nom, _, _ in paires:
        fj = os.path.join(a.dossier_resultats, f"{nom}.stats.json")
        if os.path.exists(fj):
            with open(fj, encoding="utf-8") as f:
                stats.append(json.load(f))
    if stats:
        pd.DataFrame(stats).to_csv(
            os.path.join(a.dossier_resultats, "controle_synchro.csv"), index=False)

    # ---- fusion en resultats_analyse.csv
    termines = [p[0] for p in paires if fait(p[0])]
    premier, lignes = True, 0
    it = tqdm(termines, desc="Fusion", unit="video") if tqdm else termines
    for nom in it:
        d = pd.read_csv(os.path.join(a.dossier_resultats, f"{nom}.csv"))
        participant, regardeur = decouper_nom(nom)
        d.insert(0, "regardeur", regardeur)
        d.insert(0, "participant", participant)
        d.insert(0, "video", nom)
        d.to_csv(a.sortie, mode="w" if premier else "a", header=premier, index=False)
        premier = False
        lignes += len(d)

    # ---- bilan
    print("\n================ BILAN ================")
    print(f"Videos fusionnees : {len(termines)} / {len(paires) + len(sans_video)}")
    print(f"Lignes ecrites    : {lignes}")
    if termines:
        print(f"Fichier final     : {a.sortie}")
    if sans_video:
        print(f"Sans video        : {', '.join(sans_video)}")
    if echecs:
        print(f"ECHECS            : {', '.join(echecs)}  (relancez la meme commande)")
    alertes = [s for s in stats if s["alerte"]]
    if alertes:
        print("A verifier (voir controle_synchro.csv) :")
        for s in alertes:
            print(f"  {s['video']} : {s['alerte']}")


if __name__ == "__main__":
    main()
