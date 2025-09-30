import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Robuste Pfade definieren ---
# Annahme: Dieses Skript liegt im 'src'-Ordner
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(project_root, 'data', 'processed', 'train')
test_dir = os.path.join(project_root, 'data', 'processed', 'test')
model_save_path = os.path.join(project_root, 'models', 'loki_model_v2.h5') # Version 2
plot_save_path = os.path.join(project_root, 'models', 'training_history_v2.png') # Version 2

# Sicherstellen, dass der "models"-Ordner existiert
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 1. Datensatz laden
print(f"Lade Trainingsdaten von: {train_dir}")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32
)

print(f"Lade Testdaten von: {test_dir}")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(150, 150),
    batch_size=32
)

# 2. Modell erstellen (mit Anti-Overfitting-Techniken)
model = models.Sequential([
    # --- NEU: Daten-Augmentierung als Teil des Modells ---
    layers.RandomFlip("horizontal", input_shape=(150, 150, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    # ----------------------------------------------------

    # Convolutional-Basis
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # --- NEU: Dropout zur Regularisierung ---
    layers.Dropout(0.25),
    # ----------------------------------------

    # Klassifizierungs-Teil
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    
    # --- NEU: Stärkeres Dropout vor der finalen Schicht ---
    layers.Dropout(0.5),
    # --------------------------------------------------
    
    layers.Dense(1, activation='sigmoid')
])

# 3. Modell kompilieren
# Erstelle eine Instanz des Adam-Optimierers mit einer kleineren Lernrate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # Standard ist 0.001

# Übergebe den konfigurierten Optimierer an die compile-Funktion
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# 4. Modell trainieren (mit mehr Epochen)
print("\n--- Starte optimiertes Modelltraining ---")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50  # --- NEU: Auf 50 erhöht Regulation gegen Overfitting  ---
)
print("--- Modelltraining abgeschlossen ---\n")

# 5. Modell speichern
model.save(model_save_path)
print(f"Modell erfolgreich unter {model_save_path} gespeichert.")

# 6. Visualisierung
# (Der Code für die Visualisierung bleibt exakt gleich)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig(plot_save_path)
print(f"Trainingsverlauf gespeichert unter: {plot_save_path}")
plt.show()