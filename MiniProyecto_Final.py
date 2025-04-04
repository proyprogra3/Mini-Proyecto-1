import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import tkinter as tk
from tkinter import Canvas
import numpy as np
import math

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Crear ventana de Tkinter
root = tk.Tk()
root.title("Sistema Drag & Drop con Figura de Mano")
canvas = Canvas(root, width=640, height=480)
canvas.pack()

# Función para crear coordenadas de estrella (sin rotación)
def get_star_coords(x, y):
    star_points = np.array([
        [0, -40],
        [12, -12],
        [40, -12],
        [18, 6],
        [24, 36],
        [0, 18],
        [-24, 36],
        [-18, 6],
        [-40, -12],
        [-12, -12]
    ])
    translated = (star_points + np.array([x, y])).flatten().tolist()
    return translated

# Crear objetos: estrella y cuadro
estrella_id = canvas.create_polygon(get_star_coords(150, 150), fill="gold", outline="black")
cuadro_id = canvas.create_rectangle(300, 100, 380, 180, fill="dodger blue")

# Crear puntero de mano (círculo rojo)
puntero_mano = canvas.create_oval(0, 0, 20, 20, fill="red")

# Variables de control
grabbing_estrella = False
grabbing_cuadro = False

# Detectar gesto de agarre
def is_grab_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_x, thumb_y = int(thumb_tip.x * 640), int(thumb_tip.y * 480)
    index_x, index_y = int(index_tip.x * 640), int(index_tip.y * 480)
    distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
    return distance < 40

# Verificar si el dedo índice está dentro de un objeto
def is_inside_object(x, y, coords):
    if len(coords) == 4:  # Rectángulo
        return coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]
    else:  # Polígono (estrella)
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        poly_path = np.array(points)
        return cv2.pointPolygonTest(poly_path.astype(np.int32), (x, y), False) >= 0

# Procesar cada frame
def process_frame():
    global grabbing_estrella, grabbing_cuadro

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dedo_x, dedo_y = int(index_tip.x * 640), int(index_tip.y * 480)

            # Mover el círculo que representa la mano
            canvas.coords(puntero_mano, dedo_x - 10, dedo_y - 10, dedo_x + 10, dedo_y + 10)

            if is_grab_gesture(hand_landmarks):
                # Revisar si está tocando la estrella
                estrella_coords = canvas.coords(estrella_id)
                if is_inside_object(dedo_x, dedo_y, estrella_coords):
                    grabbing_estrella = True
                    canvas.itemconfig(estrella_id, fill="lime green")
                # Revisar si está tocando el cuadro
                cuadro_coords = canvas.coords(cuadro_id)
                if is_inside_object(dedo_x, dedo_y, cuadro_coords):
                    grabbing_cuadro = True
                    canvas.itemconfig(cuadro_id, fill="orange red")
            else:
                # Soltar objetos
                if grabbing_estrella:
                    grabbing_estrella = False
                    canvas.itemconfig(estrella_id, fill="gold")
                if grabbing_cuadro:
                    grabbing_cuadro = False
                    canvas.itemconfig(cuadro_id, fill="dodger blue")

            # Mover objetos si están siendo agarrados
            if grabbing_estrella:
                nueva_estrella = get_star_coords(dedo_x, dedo_y)
                canvas.coords(estrella_id, *nueva_estrella)
            if grabbing_cuadro:
                # Mover el cuadro centrado en el dedo
                canvas.coords(cuadro_id, dedo_x - 40, dedo_y - 40, dedo_x + 40, dedo_y + 40)

    cv2.imshow('Seguimiento de Mano', frame)
    root.after(10, process_frame)

# Iniciar captura de video y procesamiento
cap = cv2.VideoCapture(0)
process_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()