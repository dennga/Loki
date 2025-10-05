import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Pfade definieren ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(project_root, 'data', 'processed', 'train')
test_dir = os.path.join(project_root, 'data', 'processed', 'test')
# NEUE VERSION FÜR DAS MODELL
model_save_path = os.path.join(project_root, 'models', 'loki_model_v3.h5') 
plot_save_path = os.path.join(project_root, 'models', 'training_history_v3.png')

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 1. Datensatz laden
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(150, 150), batch_size=32)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=(150, 150), batch_size=32)

# 2. Modell erstellen (mit Rescaling-Schicht)
model = models.Sequential([
    # --- NEU: NORMALISIERUNGS-SCHICHT ALS ERSTE SCHICHT ---
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    
    # Daten-Augmentierung
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Convolutional-Basis (input_shape wird hier entfernt)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Klassifizierungs-Teil
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 3. Modell kompilieren
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Modell trainieren
print("\n--- Starte optimiertes Modelltraining (Version 3) ---")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50 
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