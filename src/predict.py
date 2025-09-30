import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# 1. Argument Parser einrichten
# Erlaubt das Übergeben des Bildpfads über die Kommandozeile
parser = argparse.ArgumentParser(description='Predict if an image is a cat or a dog.')
parser.add_argument('image_path', type=str, help='Path to the image file to classify.')
args = parser.parse_args()

# 2. Modell laden
try:
    model_path = os.path.join('models', 'loki_model.h5')
    model = load_model(model_path)
    print("Modell erfolgreich geladen.")
except Exception as e:
    print(f"Fehler: Konnte das Modell nicht laden. Stelle sicher, dass 'loki_model.h5' im Ordner 'models' existiert.")
    print(e)
    exit()

# 3. Bild laden und vorbereiten
def prepare_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        # Fügt eine Dimension hinzu, um die Batch-Dimension zu simulieren (Modell erwartet 4D-Array)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array /= 255.0  # Normalisieren (Werte von 0-255 auf 0-1 skalieren)
        return img_array
    except Exception as e:
        print(f"Fehler beim Laden des Bildes: {e}")
        exit()

# 4. Vorhersage machen
prepped_image = prepare_image(args.image_path)
prediction = model.predict(prepped_image)
class_prob = prediction[0][0]

# 5. Ergebnis ausgeben
print("\n--- Vorhersage ---")
if class_prob > 0.5:
    print(f"Das Modell sagt mit einer Wahrscheinlichkeit von {class_prob * 100:.2f}% aus, dass es ein HUND ist. 🐶")
else:
    print(f"Das Modell sagt mit einer Wahrscheinlichkeit von {(1 - class_prob) * 100:.2f}% aus, dass es eine KATZE ist. 🐱")
print("------------------")