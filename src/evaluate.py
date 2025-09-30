import tensorflow as tf
import os

# 1. Pfade definieren
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'models', 'loki_model.h5')
test_dir = os.path.join(project_root, 'data', 'processed', 'test')

# 2. Modell und Daten laden
print(f"Lade Modell von: {model_path}")
model = tf.keras.models.load_model(model_path)

print(f"Lade Testdaten von: {test_dir}")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(150, 150),
    batch_size=32,
    shuffle=False
)

# 3. Modell bewerten
print("\nBewerte das Modell mit den Testdaten...")
loss, accuracy = model.evaluate(test_ds)

# 4. Ergebnis ausgeben
print("\n--- Evaluationsergebnis ---")
print(f"Verlust (Loss): {loss:.4f}")
print(f"Genauigkeit (Accuracy): {accuracy * 100:.2f}%")
print("---------------------------")