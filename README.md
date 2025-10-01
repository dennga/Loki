## Loki: Learning Object Classification Interface 🧠
Overview

Loki is a full-stack image classification project that demonstrates a complete machine learning workflow, from data preprocessing to interactive web deployment. It utilizes a Convolutional Neural Network (CNN) to classify images into "cat" or "dog" categories.

This project serves as a practical introduction to the world of machine learning, showcasing typical challenges and solutions such as combating overfitting and ensuring a consistent data pipeline. The final result is a modern, user-friendly web application capable of real-time predictions.

![Aktivitätsdiagramm des Loki-Projekts](Loki_UML.svg)


Backend

    Data Preprocessing: A script (preprocess.py) automates the division of raw data into training and testing sets.

    Model Training: A CNN is trained from scratch using TensorFlow and Keras (train.py). The training process was iteratively improved to minimize overfitting through data augmentation, dropout layers, and a fine-tuned learning rate.

    Consistent Normalization: A Rescaling layer is integrated directly into the model to guarantee consistent image preprocessing between training and application.

    Model Evaluation: A script (evaluate.py) assesses the performance of the final, trained model on unseen test data.

    REST API: An API created with Flask provides a /predict endpoint that processes image uploads and returns predictions in JSON format.

Frontend

    Modern User Interface: A clean, responsive frontend designed with Bootstrap 5.

    Dynamic Interaction: Image uploads and result displays are handled asynchronously via the JavaScript (fetch API) without page reloads.

    User Feedback: The application provides clear feedback during the analysis process and displays the final result, including a confidence score.

Tech Stack

    Backend: Python

    Machine Learning: TensorFlow, Keras

    Web Framework: Flask

    Frontend: HTML5, CSS3, JavaScript (ES6+)

    Styling: Bootstrap 5

    Data Processing: NumPy

Project Workflow & Insights

The project's workflow followed an iterative process typical for machine learning applications. After an initial training run, the metrics showed significant overfitting. Through targeted measures such as data augmentation, dropout regularization, and adjusting the learning rate, the model's ability to generalize was significantly improved. A further challenge was ensuring a consistent data pipeline, which was solved by integrating a Rescaling layer directly into the model.


Author

Dennis Garscha

