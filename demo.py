import cv2
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FacialRecognitionDemo:
    def __init__(self, master):
        self.master = master
        self.master.title("Demo de Reconocimiento Facial")
        self.master.geometry("800x600")

        self.video_capture = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.db_conn = sqlite3.connect('facial_data.db')
        self.create_table()

        self.setup_main_menu()

    def setup_main_menu(self):
        self.clear_window()

        self.title_label = tk.Label(self.master, text="Sistema de Reconocimiento Facial", font=("Arial", 18))
        self.title_label.pack(pady=20)

        self.add_user_button = tk.Button(self.master, text="Añadir Usuario", command=self.setup_add_user, font=("Arial", 14))
        self.add_user_button.pack(pady=10)

        self.recognize_button = tk.Button(self.master, text="Iniciar Reconocimiento", command=self.setup_recognition, font=("Arial", 14))
        self.recognize_button.pack(pady=10)

        self.quit_button = tk.Button(self.master, text="Salir", command=self.quit, font=("Arial", 14))
        self.quit_button.pack(pady=10)

    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def create_table(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT,
                           face_data BLOB)''')
        self.db_conn.commit()

    def setup_add_user(self):
        self.clear_window()
        self.video_capture = cv2.VideoCapture(0)

        self.video_frame = tk.Label(self.master)
        self.video_frame.pack(pady=10)

        self.name_label = tk.Label(self.master, text="Nombre:", font=("Arial", 12))
        self.name_label.pack()

        self.name_entry = tk.Entry(self.master, font=("Arial", 12))
        self.name_entry.pack()

        self.capture_button = tk.Button(self.master, text="Capturar", command=self.capture_face, font=("Arial", 12))
        self.capture_button.pack(pady=10)

        self.back_button = tk.Button(self.master, text="Volver", command=self.return_to_main, font=("Arial", 12))
        self.back_button.pack(pady=10)

        self.update_video()

    def update_video(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            self.video_frame.after(10, self.update_video)

    def capture_face(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Por favor, ingrese un nombre")
            return

        ret, frame = self.video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            messagebox.showerror("Error", "No se detectó ningún rostro")
            return

        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_blob = cv2.imencode('.png', face_resized)[1].tobytes()

        cursor = self.db_conn.cursor()
        cursor.execute("INSERT INTO faces (name, face_data) VALUES (?, ?)", (name, face_blob))
        self.db_conn.commit()

        messagebox.showinfo("Éxito", f"Rostro de {name} capturado y almacenado")

    def setup_recognition(self):
        self.clear_window()
        self.video_capture = cv2.VideoCapture(0)
        self.known_faces, self.known_names = self.load_known_faces()

        self.video_frame = tk.Label(self.master)
        self.video_frame.pack(pady=10)

        self.info_label = tk.Label(self.master, text="Esperando reconocimiento...", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.back_button = tk.Button(self.master, text="Volver", command=self.return_to_main, font=("Arial", 12))
        self.back_button.pack(pady=10)

        self.update_recognition()

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

    def update_recognition(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (128, 128))

                    min_dist = float('inf')
                    recognized_name = "Desconocido"

                    for i, known_face in enumerate(self.known_faces):
                        dist = np.linalg.norm(face_resized.flatten() - known_face.flatten())
                        if dist < min_dist:
                            min_dist = dist
                            recognized_name = self.known_names[i]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    self.info_label.config(text=f"Usuario reconocido: {recognized_name}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

            self.video_frame.after(10, self.update_recognition)

    def return_to_main(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.setup_main_menu()

    def quit(self):
        if self.video_capture is not None:
            self.video_capture.release()
        self.db_conn.close()
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionDemo(root)
    root.mainloop()