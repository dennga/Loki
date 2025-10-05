document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const resultElement = document.getElementById('result');
    const imageInput = document.getElementById('image-input');

    form.addEventListener('submit', async (event) => {
        // Verhindert das Neuladen der Seite
        event.preventDefault();

        // Zeige einen Lade-Status an
        resultElement.textContent = 'Analysiere...';

        // Erstelle FormData-Objekt aus dem Formular
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);

        try {
            // Sende die Daten im Hintergrund an den /predict Endpunkt
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            // Wandle die JSON-Antwort vom Server um
            const data = await response.json();

            // Zeige das Ergebnis an
            if (data.error) {
                resultElement.textContent = `Fehler: ${data.error}`;
            } else {
                resultElement.textContent = `${data.prediction} (${data.confidence}%)`;
            }

        } catch (error) {
            resultElement.textContent = 'Ein Fehler ist aufgetreten. Bitte versuche es erneut.';
            console.error('Fehler:', error);
        } finally {
            // Setzt das Formular zurück und leert damit auch das Dateifeld.
            form.reset();
        }
    });
});