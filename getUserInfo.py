import cv2
import os
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FacialDataCollector:
    def __init__(self, master):
        self.master = master
        self.master.title("Recolector de Datos Faciales")
        self.master.geometry("800x600")

        self.video_capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.db_conn = sqlite3.connect('facial_data.db')
        self.create_table()

        self.setup_gui()

    def setup_gui(self):
        self.video_frame = tk.Label(self.master)
        self.video_frame.pack(pady=10)

        self.name_label = tk.Label(self.master, text="Nombre:")
        self.name_label.pack()

        self.name_entry = tk.Entry(self.master)
        self.name_entry.pack()

        self.capture_button = tk.Button(self.master, text="Capturar", command=self.capture_face)
        self.capture_button.pack(pady=10)

        self.quit_button = tk.Button(self.master, text="Salir", command=self.quit)
        self.quit_button.pack(pady=10)

        self.update_video()

    def create_table(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT,
                           face_data BLOB)''')
        self.db_conn.commit()

    def update_video(self):
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

    def quit(self):
        self.video_capture.release()
        self.db_conn.close()
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialDataCollector(root)
    root.mainloop()