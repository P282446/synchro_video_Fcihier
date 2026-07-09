from pathlib import Path



def parcours_dossier(chemin_dossier):  
    dossier = Path(chemin_dossier)
    t = []
    v = []   
    for sous_dossier in dossier.rglob("*"):
        if sous_dossier.is_dir():
    
            fichiers_txt = list(sous_dossier.glob("*.txt"))
            fichiers_video = list(sous_dossier.glob("*.wmv"))   # ou *.mp4
    
            if fichiers_txt and fichiers_video:
                #print(f"Dossier : {sous_dossier}")
    
                for txt in fichiers_txt:
                    #print("  TXT   :", txt)
                    t.append(txt)
    
                for video in fichiers_video:
                    #print("  VIDEO :", video)
                    v.append(video)                   
    return t, v