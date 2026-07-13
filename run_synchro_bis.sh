#!/bin/bash

VIDEO_DIR=~/Bureau/videos
TXT_DIR=~/Bureau/fichiers
OUTPUT_DIR=~/Bureau/resultat_synchro
SCRIPT_DIR=~/Bureau/synchronisation/synchro_video_Fcihier

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR" || exit 1

for video_path in "$VIDEO_DIR"/seg_*.wmv; do
    filename=$(basename "$video_path")
    base_name="${filename#seg_}"      # retire le préfixe "seg_"
    base_name="${base_name%.wmv}"      # retire l'extension ".wmv"

    txt_path="$TXT_DIR/${base_name}.txt"

    echo "=================================================="
    echo "Vidéo   : $video_path"
    echo "Fichier : $txt_path"
    echo "=================================================="

    if [ ! -f "$txt_path" ]; then
        echo "⚠️  Fichier .txt introuvable pour $base_name, on saute."
        continue
    fi

    python synchro_bis.py "$video_path" "$txt_path" -o "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✅ Terminé : $base_name"
    else
        echo "❌ Échec pour : $base_name"
    fi
done

echo "=================================================="
echo "Traitement terminé pour toutes les vidéos."
echo "=================================================="
