import os
import shutil
import random

"""This script processes and organizes the raw data for model training."""

# 1. Robuste Pfade definieren
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')

# 2. Alten Ordner löschen für sauberen Start
print(f"Überprüfe und bereinige '{PROCESSED_DATA_DIR}'...")
if os.path.exists(PROCESSED_DATA_DIR):
    shutil.rmtree(PROCESSED_DATA_DIR)

# 3. Neue Ordnerstruktur erstellen
print("Erstelle neue Ordnerstruktur...")
train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')
os.makedirs(os.path.join(train_dir, 'cats'))
os.makedirs(os.path.join(train_dir, 'dogs'))
os.makedirs(os.path.join(test_dir, 'cats'))
os.makedirs(os.path.join(test_dir, 'dogs'))

# 4. Dateipfade sammeln
cat_files = [os.path.join(RAW_DATA_DIR, 'cats', f) for f in os.listdir(os.path.join(RAW_DATA_DIR, 'cats'))]
dog_files = [os.path.join(RAW_DATA_DIR, 'dogs', f) for f in os.listdir(os.path.join(RAW_DATA_DIR, 'dogs'))]

# 5. Daten mischen und aufteilen
random.shuffle(cat_files)
random.shuffle(dog_files)
split_ratio = 0.8
train_cats = cat_files[:int(len(cat_files) * split_ratio)]
test_cats = cat_files[int(len(cat_files) * split_ratio):]
train_dogs = dog_files[:int(len(dog_files) * split_ratio)]
test_dogs = dog_files[int(len(dog_files) * split_ratio):]

# 6. Helferfunktion zum Kopieren
def copy_files(file_list, destination_folder):
    print(f"Kopiere {len(file_list)} Dateien nach '{destination_folder}'...")
    for file_path in file_list:
        shutil.copy(file_path, destination_folder)

# 7. Dateien kopieren
copy_files(train_cats, os.path.join(train_dir, 'cats'))
copy_files(train_dogs, os.path.join(train_dir, 'dogs'))
copy_files(test_cats, os.path.join(test_dir, 'cats'))
copy_files(test_dogs, os.path.join(test_dir, 'dogs'))

print("\nDatenverarbeitung erfolgreich abgeschlossen!")

