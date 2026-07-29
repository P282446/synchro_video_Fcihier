#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ce programme effectue la synchronisation entre une vidéo segmentée et un
fichier .txt issu du fichier .eta (coordonnées de regard).

La fonction centrale de ce programme est classification_pixels.classification_with_coor(video_path, coor_eye).

Entrées :
    video_path : chemin de la vidéo segmentée
    file       : chemin du fichier .txt contenant les coordonnées du regard
    output_dir : (optionnel) dossier de sortie du CSV résultat
                 par défaut, même dossier que le fichier .txt

Sortie :
    Un fichier CSV {base_name}_result.csv contenant les données d'entrée
    enrichies d'une colonne "region".
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

import classification_pixels as cp
import Parcours_dossier as pcd  # noqa: F401 (conservé si utilisé ailleurs dans le pipeline)
from typing import Optional

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Synchronise une vidéo segmentée avec un fichier de "
                    "coordonnées de regard (.txt) et produit un CSV annoté "
                    "par zone."
    )
    parser.add_argument("video_path", help="Chemin de la vidéo segmentée (.wmv/.avi/...)")
    parser.add_argument("gaze_file", help="Chemin du fichier .txt de coordonnées de regard")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Dossier de sortie du CSV résultat "
             "(par défaut : même dossier que le fichier .txt)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Active les logs détaillés (DEBUG)"
    )
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_inputs(video_path: str, gaze_file: str):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")
    if not os.path.isfile(gaze_file):
        raise FileNotFoundError(f"Fichier de coordonnées introuvable : {gaze_file}")


def load_gaze_data(gaze_file: str) -> pd.DataFrame:
    # Détection automatique du séparateur (tab ou virgule)
    with open(gaze_file, 'r') as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(gaze_file, sep=sep)

    # Normalise les noms de colonnes : remplace les points par des espaces
    # au cas où le fichier vient d'un export R (qui convertit espaces -> points)
    df.columns = [col.replace(".", " ") for col in df.columns]

    required_cols = {"Lft X Pos", "Lft Y Pos"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {gaze_file} : {missing}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    return df


def build_output_path(gaze_file: str, output_dir: Optional[str]) -> str:
    base_name = os.path.splitext(os.path.basename(gaze_file))[0]
    target_dir = output_dir if output_dir else os.path.dirname(gaze_file) or "."
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, base_name + "_result_synchro.csv")

def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        validate_inputs(args.video_path, args.gaze_file)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    output_path = build_output_path(args.gaze_file, args.output_dir)

    logging.info("Vidéo   : %s", args.video_path)
    logging.info("Fichier : %s", args.gaze_file)
    logging.info("Sortie  : %s", output_path)

    try:
        gaze_df = load_gaze_data(args.gaze_file)
    except (ValueError, pd.errors.ParserError) as e:
        logging.error("Erreur de lecture du fichier de coordonnées : %s", e)
        sys.exit(1)

    coor_eye = gaze_df[["Lft X Pos", "Lft Y Pos"]]


    logging.info("Démarrage de la classification...")
    start = time.time()
    try:
        zones = cp.classification_with_coor(args.video_path, coor_eye)
    except Exception as e:
        logging.error("Échec de la classification : %s", e)
        sys.exit(1)
    elapsed = time.time() - start
    logging.info("Temps d'exécution : %.2f minutes", elapsed / 60)

    if "zones" not in zones:
        logging.error("Sortie inattendue de classification_with_coor : "
                       "clé 'zones' absente.")
        sys.exit(1)

    gaze_df["region"] = zones["zones"]

    try:
        gaze_df.to_csv(output_path, index=False)
    except OSError as e:
        logging.error("Impossible d'écrire le fichier de sortie %s : %s", output_path, e)
        sys.exit(1)

    logging.info("✅ Synchronisation terminée : %s", output_path)


if __name__ == "__main__":
    main()
