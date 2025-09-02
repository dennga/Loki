# Loki: Learning Object Klassification Interface 🧠

Übersicht

Loki ist ein einfaches, aber funktionelles Projekt zur Bildklassifizierung. Es nutzt ein Convolutional Neural Network (CNN), um zu lernen, Bilder zu erkennen und zu klassifizieren. Dieses Projekt dient als praktischer Einstieg in die Welt des maschinellen Lernens und der künstlichen Intelligenz, insbesondere im Bereich der Computer Vision.

Features

    Datensatzvorbereitung: Automatisiert das Laden und Vorverarbeiten von Bilddaten.

    Modelltraining: Erstellt und trainiert ein neuronales Netz für die Bildklassifizierung.

    Vorhersage: Kann neue, unbekannte Bilder klassifizieren.

    Metriken: Zeigt die Genauigkeit und den Verlust des trainierten Modells an.

Installation

Stelle sicher, dass Python 3.x auf deinem System installiert ist. Klone dann dieses Repository und installiere die notwendigen Abhängigkeiten:
Bash

git clone https://github.com/DEIN-BENUTZERNAME/loki.git
cd loki
pip install -r requirements.txt

Wichtiger Hinweis

Die requirements.txt-Datei muss von dir selbst erstellt werden. Hier sind die grundlegenden Pakete, die du benötigst:

tensorflow
keras
numpy
matplotlib

Du kannst die Datei mit dem Befehl pip freeze > requirements.txt generieren, wenn du alle Pakete installiert hast.

Benutzung

    Datensatz vorbereiten: Lade den Datensatz "Cat vs. Dog" herunter und organisiere die Bilder in den Ordnern train/cats und train/dogs.

    Modell trainieren: Führe das Trainingsskript aus. Dies wird das neuronale Netz trainieren und das trainierte Modell speichern.
    Bash

python train.py

Bilder vorhersagen: Nutze das Vorhersageskript, um neue Bilder zu klassifizieren.
Bash

    python predict.py --image_path path/to/dein_bild.jpg

Beiträge

Jede Art von Beitrag ist willkommen! Wenn du neue Funktionen vorschlagen oder Fehler melden möchtest, nutze bitte die GitHub Issues.

Lizenz

Dieses Projekt steht unter der MIT-Lizenz.

Autor

Dein Name oder GitHub-Benutzername
