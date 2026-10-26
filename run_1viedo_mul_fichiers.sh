#!/bin/bash
#
# synchro_batch.sh
#
# Lance synchro_bis.py pour plusieurs fichiers (CSV) sur une meme video.
#
# Usage :
#   ./synchro_batch.sh chemin_video.wmv fichier1.csv fichier2.csv ...
#   ./synchro_batch.sh chemin_video.wmv --dir dossier_contenant_csv/
#
# Exemple concret (equivalent a tes 3 commandes manuelles) :
#   ./synchro_batch.sh ~/Bureau/videos/seg_CEG-047-Psy.wmv \
#       ~/Bureau/etude_decalage/CEG.047-Psy_LM_v2.csv \
#       ~/Bureau/etude_decalage/CEG.047-Psy_GAM.csv \
#       ~/Bureau/etude_decalage/CEG.047-Psy_MIXTE.csv

set -uo pipefail

SCRIPT_PYTHON="synchro_bis.py"   # nom/chemin du script python a appeler
LOG_DIR="./logs_synchro"
mkdir -p "$LOG_DIR"

horodatage() {
    date '+%Y-%m-%d %H:%M:%S'
}

if [ $# -lt 2 ]; then
    echo "Usage : $0 chemin_video.wmv fichier1.csv [fichier2.csv ...]"
    echo "        $0 chemin_video.wmv --dir dossier_contenant_csv/"
    exit 1
fi

VIDEO="$1"
shift

if [ ! -f "$VIDEO" ]; then
    echo "❌ Vidéo introuvable : $VIDEO"
    exit 1
fi

# Construction de la liste des fichiers CSV a traiter
FICHIERS=()

if [ "$1" == "--dir" ]; then
    DOSSIER="$2"
    if [ ! -d "$DOSSIER" ]; then
        echo "❌ Dossier introuvable : $DOSSIER"
        exit 1
    fi
    while IFS= read -r -d '' fichier; do
        FICHIERS+=("$fichier")
    done < <(find "$DOSSIER" -maxdepth 1 -type f -name "*.csv" -print0 | sort -z)
else
    FICHIERS=("$@")
fi

if [ ${#FICHIERS[@]} -eq 0 ]; then
    echo "❌ Aucun fichier CSV a traiter."
    exit 1
fi

NB_TOTAL=${#FICHIERS[@]}
NB_OK=0
NB_ECHEC=0
DEBUT_GLOBAL=$(date +%s)

echo "=================================================="
echo "Vidéo utilisée : $VIDEO"
echo "Nombre de fichiers à synchroniser : $NB_TOTAL"
echo "=================================================="

for i in "${!FICHIERS[@]}"; do
    FICHIER="${FICHIERS[$i]}"
    NUM=$((i + 1))

    if [ ! -f "$FICHIER" ]; then
        echo "⚠️  [$NUM/$NB_TOTAL] Fichier introuvable, ignoré : $FICHIER"
        NB_ECHEC=$((NB_ECHEC + 1))
        continue
    fi

    NOM_BASE=$(basename "$FICHIER")
    LOG_FICHIER="$LOG_DIR/${NOM_BASE%.csv}.log"

    echo ""
    echo "=================================================="
    echo "[$NUM/$NB_TOTAL] $(horodatage) Traitement de : $FICHIER"
    echo "=================================================="

    DEBUT=$(date +%s)

    python3 "$SCRIPT_PYTHON" "$VIDEO" "$FICHIER" 2>&1 | tee "$LOG_FICHIER"
    CODE_RETOUR=${PIPESTATUS[0]}

    FIN=$(date +%s)
    DUREE=$(( (FIN - DEBUT) / 60 ))

    if [ "$CODE_RETOUR" -eq 0 ]; then
        echo "✅ [$NUM/$NB_TOTAL] Succès pour $NOM_BASE (${DUREE} min) — log : $LOG_FICHIER"
        NB_OK=$((NB_OK + 1))
    else
        echo "❌ [$NUM/$NB_TOTAL] Échec pour $NOM_BASE (code $CODE_RETOUR) — voir log : $LOG_FICHIER"
        NB_ECHEC=$((NB_ECHEC + 1))
    fi
done

FIN_GLOBAL=$(date +%s)
DUREE_GLOBALE=$(( (FIN_GLOBAL - DEBUT_GLOBAL) / 60 ))

echo ""
echo "=================================================="
echo "Résumé de la synchronisation par lot"
echo "=================================================="
echo "Réussites : $NB_OK / $NB_TOTAL"
echo "Échecs    : $NB_ECHEC / $NB_TOTAL"
echo "Durée totale : ${DUREE_GLOBALE} min"
echo "Logs disponibles dans : $LOG_DIR/"
echo "=================================================="
