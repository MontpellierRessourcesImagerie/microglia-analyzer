import os
import shutil
import unicodedata
import sys

def normalize_filename(filename):
    # Remplace les accents et autres caractères spéciaux
    normalized = unicodedata.normalize('NFD', filename)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Remplace les espaces par des tirets
    normalized = normalized.replace(" ", "-")
    
    # Transforme le nom de fichier en minuscules
    normalized = normalized.lower()
    
    return normalized

def copy_and_rename_files(src_folder, dest_folder):
    # Vérifie si les dossiers source et destination existent
    if not os.path.isdir(src_folder):
        print(f"Le dossier source '{src_folder}' n'existe pas.")
        return
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)  # Crée le dossier de destination s'il n'existe pas encore

    # Parcours des fichiers dans le dossier source
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        
        # Vérifie si c'est un fichier
        if os.path.isfile(src_file):
            # Renomme le fichier selon les règles spécifiées
            new_filename = normalize_filename(filename)
            dest_file = os.path.join(dest_folder, new_filename)
            
            # Copie le fichier vers le dossier de destination
            shutil.copy2(src_file, dest_file)
            print(f"Copié : '{filename}' -> '{new_filename}'")

if __name__ == "__main__":

    src_folder = "/home/benedetti/Documents/projects/2060-microglia/data/input_masks/"
    dest_folder = "/home/benedetti/Documents/projects/2060-microglia/data/08-patches-no-spe-char-masks/"
    
    copy_and_rename_files(src_folder, dest_folder)
