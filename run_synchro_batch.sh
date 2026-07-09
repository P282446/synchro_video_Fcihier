#!/bin/bash

VIDEO_DIR=~/Bureau/videos
TXT_DIR=~/Bureau/fichiers
OUTPUT_DIR=~/Bureau/resultat_synchro
SCRIPT_DIR=~/Bureau/synchronisation/synchro_video_Fcihier

mkdir -p "$OUTPUT_DIR"

for video_path in "$VIDEO_DIR"/seg_*.wmv; do
    filename=$(basename "$video_path")
    base_name="${filename#seg_}"      # retire le préfixe "seg_"
    base_name="${base_name%.wmv}"      # retire l'extension ".wmv"

    txt_path="$TXT_DIR/${base_name}.txt"
    generated_csv="$TXT_DIR/${base_name}_result.csv"
    final_csv="$OUTPUT_DIR/${base_name}_result.csv"

    echo "=================================================="
    echo "Vidéo   : $video_path"
    echo "Fichier : $txt_path"
    echo "=================================================="

    if [ ! -f "$txt_path" ]; then
        echo "⚠️  Fichier .txt introuvable pour $base_name, on saute."
        continue
    fi

    cd "$SCRIPT_DIR" || exit 1
    python synchronisation.py "$video_path" "$txt_path"

    if [ $? -eq 0 ] && [ -f "$generated_csv" ]; then
        mv "$generated_csv" "$final_csv"
        echo "✅ Terminé : $base_name → $final_csv"
    else
        echo "❌ Échec pour : $base_name"
    fi
done
