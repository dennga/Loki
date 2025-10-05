import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Initialisierung der App und Konfiguration ---
app = Flask(__name__)

# Pfade definieren
project_root = os.path.dirname(os.path.abspath(__file__))
# Lade das neue V3 Modell
model_path = os.path.join(project_root, 'models', 'loki_model_v3.h5')
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Modell laden (nur einmal beim Start) ---
try:
    model = load_model(model_path)
    print(f"Modell von {model_path} erfolgreich geladen.")
except Exception as e:
    print(f"FEHLER: Modell konnte nicht geladen werden. Stelle sicher, dass die Datei existiert.")
    print(e)
    model = None

# --- Helferfunktion zur Bildvorbereitung (ohne manuelle Normalisierung) ---
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
        print(f"Fehler bei der Bildvorbereitung: {e}")
        return None

# --- Routen der Web-Anwendung ---

@app.route('/')
def home():
    """Rendert die Hauptseite (index.html)."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Nimmt ein Bild entgegen, macht eine Vorhersage und gibt JSON zurück."""
    if model is None:
        return jsonify({'error': 'Modell ist nicht geladen, bitte Server-Log prüfen.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Kein Bild in der Anfrage gefunden.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Keine Datei ausgewählt.'}), 400

    if file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            prepped_image = prepare_image(filepath)
            
            if prepped_image is None:
                return jsonify({'error': 'Bild konnte nicht verarbeitet werden.'}), 400

            prediction = model.predict(prepped_image)
            class_prob = prediction[0][0]

            print(f"DEBUG: Rohe Vorhersage (Wahrscheinlichkeit für Hund) = {class_prob}")

            if class_prob > 0.5:
                prediction_label = "Hund 🐶"
                confidence = class_prob * 100
            else:
                prediction_label = "Katze 🐱"
                confidence = (1 - class_prob) * 100

            os.remove(filepath)

            return jsonify({
                'prediction': prediction_label,
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            print(f"Ein Fehler ist im /predict Endpunkt aufgetreten: {e}")
            return jsonify({'error': 'Ein interner Serverfehler ist aufgetreten.'}), 500

# --- Startpunkt der Anwendung ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
