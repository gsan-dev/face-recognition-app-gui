import cv2
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FacialRecognitionSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Sistema de Reconocimiento Facial")
        self.master.geometry("800x600")

        self.video_capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.db_conn = sqlite3.connect('facial_data.db')
        self.known_faces, self.known_names = self.load_known_faces()

        self.setup_gui()

    def setup_gui(self):
        self.video_frame = tk.Label(self.master)
        self.video_frame.pack(pady=10)

        self.info_label = tk.Label(self.master, text="Esperando reconocimiento...", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.quit_button = tk.Button(self.master, text="Salir", command=self.quit)
        self.quit_button.pack(pady=10)

        self.update_video()

    def load_known_faces(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name, face_data FROM faces")
        results = cursor.fetchall()

        known_faces = []
        known_names = []

        for name, face_data in results:
            nparr = np.frombuffer(face_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            known_faces.append(img)
            known_names.append(name)

        return known_faces, known_names

    def update_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (128, 128))

                # Realizar reconocimiento facial
                min_dist = float('inf')
                recognized_name = "Desconocido"

                for i, known_face in enumerate(self.known_faces):
                    dist = np.linalg.norm(face_resized.flatten() - known_face.flatten())
                    if dist < min_dist:
                        min_dist = dist
                        recognized_name = self.known_names[i]

                # Dibujar rectángulo y nombre
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.info_label.config(text=f"Usuario reconocido: {recognized_name}")

            # Convertir el frame para mostrarlo en tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.video_frame.after(10, self.update_video)

    def quit(self):
        self.video_capture.release()
        self.db_conn.close()
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionSystem(root)
    root.mainloop()