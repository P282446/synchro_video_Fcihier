# Pipeline de synchronisation vidéo et données oculométriques

## 1. Présentation du projet

Ce pipeline permet de synchroniser une vidéo segmentée avec un fichier de coordonnées oculométriques issu d'un dispositif d'eye-tracking.

Les fichiers de coordonnées sont au format `.txt`, avec des champs séparés par des tabulations.

Le pipeline propose deux modes d'utilisation :
- **Mode individuel** : synchronisation d'une seule vidéo avec un fichier de coordonnées.
- **Mode batch** : synchronisation de plusieurs vidéos et fichiers de coordonnées à partir de dossiers.

## 2. Prérequis

Avant d'utiliser le pipeline, les logiciels suivants doivent être disponibles sur votre système :

- Python 3
- `pip`, le gestionnaire de paquets Python
- Bash, pour l'exécution du script batch

## 3. Installation

### 3.1. Cloner le dépôt GitLab

Clonez le dépôt, puis accédez au répertoire du projet :

```bash
git clone git@github.com:P282446/synchro_video_Fcihier.git
cd synchro_video_fichier
```

Remplacez `git@github.com:P282446/synchro_video_Fcihier.git` et `synchro_video_fichier` par les informations de votre dépôt.

### 3.2. Créer un environnement virtuel

Créez un environnement virtuel Python :

```bash
python3 -m venv .venv
```

Activez-le :

```bash
source .venv/bin/activate
```

Une fois l'environnement activé, le préfixe `(.venv)` apparaît généralement dans le terminal.

### 3.3. Installer les dépendances

Installez les bibliothèques nécessaires au fonctionnement du pipeline :

```bash
python -m pip install --upgrade pip
python -m pip install tqdm pandas numpy opencv-python
```

Ces bibliothèques permettent notamment de manipuler les données tabulaires, de traiter les images vidéo et de suivre la progression des traitements.

## 4. Utilisation du pipeline

### 4.1. Synchroniser une seule vidéo

Pour synchroniser une vidéo avec son fichier de coordonnées, exécutez :

```bash
python3 synchro_bis.py \
    chemin/vers/video \
    chemin/vers/fichier_coordonnees
```

Remplacez les chemins d'exemple par les chemins réels des fichiers à traiter.

**Entrées :**
- Une vidéo déjà segmentée.
- Un fichier de coordonnées oculométriques au format `.txt`, avec des tabulations comme séparateurs.

**Sortie :**

Le résultat est produit selon les paramètres et le chemin de sortie définis dans le script `synchro_bis.py`.

### 4.2. Synchroniser plusieurs vidéos (mode batch)

Le mode batch permet de traiter un ensemble de vidéos et de fichiers de coordonnées en exécutant le script `run_synchro_bis.sh`.

#### Étape 1 : Organiser les fichiers

Regroupez les vidéos dans un même dossier et les fichiers de coordonnées correspondants dans un autre dossier.

#### Étape 2 : Configurer le script batch

Ouvrez `run_synchro_bis.sh` et adaptez les chemins afin d'indiquer :

- Le dossier contenant les vidéos.
- Le dossier contenant les fichiers de coordonnées.
- Le dossier de destination des résultats.

Vérifiez également que l'association entre chaque vidéo et son fichier de coordonnées correspondants est correctement définie dans le script.

#### Étape 3 : Exécuter le traitement

Depuis la racine du projet, lancez :

```bash
bash run_synchro_bis.sh
```

Le script lance successivement les synchronisations prévues par sa configuration.

## 5. Structure du projet

```text
.
├── synchro_bis.py          # Synchronisation d'une vidéo et de ses coordonnées
├── run_synchro_bis.sh      # Exécution en mode batch
├── .venv/                  # Environnement virtuel Python
└── README.md               # Documentation du projet
```

Cette structure est indicative et peut être adaptée à l'organisation réelle du dépôt.

## 6. Dépannage

**Erreur : module Python introuvable**

Vérifiez que l'environnement virtuel est activé et que les dépendances ont été installées :

```bash
source .venv/bin/activate
python -m pip install tqdm pandas numpy opencv-python
```

**Erreur : fichier introuvable**

Vérifiez les chemins des vidéos et des fichiers de coordonnées, ainsi que l'orthographe de leurs noms.

**Erreur lors du traitement batch**

Vérifiez les chemins définis dans `run_synchro_bis.sh` et assurez-vous que les dossiers d'entrée existent et contiennent les fichiers attendus.

## 7. Remarques

- Les vidéos doivent être segmentées avant d'être traitées par ce pipeline.
- Les fichiers de coordonnées doivent respecter le format attendu par le programme.
- Il est recommandé de tester d'abord le traitement sur une seule vidéo avant de lancer un traitement batch.
- Conservez les fichiers sources afin de pouvoir reproduire les traitements et vérifier les résultats.


