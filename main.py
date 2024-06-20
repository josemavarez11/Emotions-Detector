import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Mapeo de índices a nombres de emociones
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprised",
    6: "Neutral"
}

# Cargar el modelo entrenado
model = load_model('emotion_detector_model.h5')

root = tk.Tk()
root.title("Emotions Detector")

# Referencias a widgets actuales de imagen y etiqueta de emoción
current_image_panel = None
current_emotion_label = None

def cargar_imagen():
    global current_image_panel, current_emotion_label
    
    file_path = filedialog.askopenfilename()
    if file_path:
        # Quitar la imagen y la etiqueta de emoción anteriores si existen
        if current_image_panel is not None:
            current_image_panel.pack_forget()
            current_image_panel.destroy()
        if current_emotion_label is not None:
            current_emotion_label.pack_forget()
            current_emotion_label.destroy()
        
        img = Image.open(file_path)
        img = img.resize((48, 48), Image.LANCZOS).convert('L')
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        emotion_index = np.argmax(pred)
        emotion = emotion_dict[emotion_index]

        # Muestra la imagen y la emoción detectada
        img_display = Image.open(file_path)
        img_display = img_display.resize((250, 250), Image.LANCZOS)
        img_display = ImageTk.PhotoImage(img_display)
        current_image_panel = tk.Label(root, image=img_display)
        current_image_panel.image = img_display
        current_image_panel.pack()

        current_emotion_label = tk.Label(root, text=f"Emotion detected: {emotion}")
        current_emotion_label.pack()

def activar_camara():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            pred = model.predict(roi_gray)
            emotion_index = np.argmax(pred)
            emotion = emotion_dict[emotion_index]
            emotion_text = f"Emotion: {emotion}"

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Emotion Detections', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

btn_cargar_imagen = tk.Button(root, text="Load Image", command=cargar_imagen)
btn_cargar_imagen.pack(pady=10)

btn_activar_camara = tk.Button(root, text="Activate Camera", command=activar_camara)
btn_activar_camara.pack(pady=10)

root.mainloop()
