import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# 1. Pfade und Argumente
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Lade das neue V3 Modell
model_path = os.path.join(project_root, 'models', 'loki_model_v3.h5')

parser = argparse.ArgumentParser(description='Predict if an image is a cat or a dog.')
parser.add_argument('image_path', type=str, help='Path to the image file to classify.')
args = parser.parse_args()

# 2. Modell laden
try:
    model = load_model(model_path)
    print(f"Modell von {model_path} geladen.")
except Exception as e:
    print(f"Fehler: Modell konnte nicht geladen werden. {e}")
    exit()

# 3. Bild vorbereiten (ohne manuelle Normalisierung)
def prepare_image(image_path):
    """Lädt ein Bild und skaliert es auf 150x150."""
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Die Normalisierung (img_array /= 255.0) wurde entfernt,
        # da sie jetzt im Modell (Rescaling-Layer) stattfindet.
        return img_array
    except Exception as e:
        print(f"Fehler beim Laden des Bildes: {e}")
        exit()

# 4. Vorhersage machen
prepped_image = prepare_image(args.image_path)
prediction = model.predict(prepped_image)
class_prob = prediction[0][0]

print(f"DEBUG: Rohe Vorhersage (Wahrscheinlichkeit für Hund) = {class_prob}")

# 5. Ergebnis ausgeben
print("\n--- Vorhersage ---")
if class_prob > 0.5:
    print(f"Das Modell ist zu {class_prob * 100:.2f}% sicher, dass es ein HUND ist. 🐶")
else:
    print(f"Das Modell ist zu {(1 - class_prob) * 100:.2f}% sicher, dass es eine KATZE ist. 🐱")
print("------------------")
