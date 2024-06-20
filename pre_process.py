import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_fer2013():
    data = pd.read_csv('fer2013.csv')
    width, height = 48, 48
    faces = []
    emotions = []

    for index, row in data.iterrows():
        pixels = np.asarray(row['pixels'].split(' '), dtype=np.uint8).reshape(width, height)
        faces.append(pixels)
        emotions.append(row['emotion'])

    faces = np.asarray(faces)
    emotions = np.asarray(emotions)
    
    return faces, emotions

def preprocess_data(faces, emotions):
    # Normalizar las imágenes
    faces = faces / 255.0
    faces = np.expand_dims(faces, -1)
    
    # Dividir en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

faces, emotions = load_fer2013()
X_train, X_val, y_train, y_val = preprocess_data(faces, emotions)
