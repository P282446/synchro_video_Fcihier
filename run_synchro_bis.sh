#!/bin/bash

VIDEO_DIR=~/Bureau/videos
TXT_DIR=~/Bureau/fichiers
OUTPUT_DIR=~/Bureau/resultat_synchro
SCRIPT_DIR=~/Bureau/synchronisation/synchro_video_Fcihier

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR" || exit 1

nb_traites=0
nb_deja_faits=0
nb_echecs=0

for video_path in "$VIDEO_DIR"/seg_*.wmv; do
    filename=$(basename "$video_path")
    base_name="${filename#seg_}"      # retire le préfixe "seg_"
    base_name="${base_name%.wmv}"      # retire l'extension ".wmv"

    txt_path="$TXT_DIR/${base_name}.txt"
    output_csv="$OUTPUT_DIR/${base_name}_result_synchro.csv"

    echo "=================================================="
    echo "Vidéo   : $video_path"
    echo "Fichier : $txt_path"
    echo "=================================================="

    # Ignore si le fichier a déjà été traité (CSV de sortie déjà présent)
    if [ -f "$output_csv" ]; then
        echo "⏭️  Déjà traité, on saute : $output_csv"
        nb_deja_faits=$((nb_deja_faits + 1))
        continue
    fi

    if [ ! -f "$txt_path" ]; then
        echo "⚠️  Fichier .txt introuvable pour $base_name, on saute."
        continue
    fi

    python synchro_bis.py "$video_path" "$txt_path" -o "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✅ Terminé : $base_name"
        nb_traites=$((nb_traites + 1))
    else
        echo "❌ Échec pour : $base_name"
        nb_echecs=$((nb_echecs + 1))
    fi
done

echo "=================================================="
echo "Traitement terminé pour toutes les vidéos."
echo "Nouveaux traitements : $nb_traites"
echo "Déjà traités (ignorés) : $nb_deja_faits"
echo "Échecs : $nb_echecs"
echo "=================================================="
