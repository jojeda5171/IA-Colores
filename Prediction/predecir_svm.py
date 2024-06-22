import os
import cv2
import numpy as np
from joblib import load
import tkinter as tk
from tkinter import filedialog, messagebox

# Función para predecir la clase de la imagen seleccionada


def predecir_imagen():
    # Obtener la ruta de la imagen seleccionada
    ruta_imagen = filedialog.askopenfilename(
        filetypes=[("Imagenes", "*.jpg *.jpeg *.png")])

    if ruta_imagen:
        # Leer la imagen a color
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

        # Cambiar tamaño de la imagen y aplicar filtro de mediana (mismo preprocesamiento que en entrenamiento)
        imagen = cv2.resize(imagen, (50, 50), interpolation=cv2.INTER_LINEAR)
        imagen = cv2.medianBlur(imagen, 3)

        # Normalizar los valores de píxeles
        imagen = imagen / 255.0

        # Flattening para convertir la imagen en un vector
        matriz_flattened = imagen.flatten()

        # Realizar la predicción utilizando el modelo cargado
        prediccion = modelo_guardado.predict([matriz_flattened])[0]

        # Mostrar la predicción en una notificación
        messagebox.showinfo("Predicción", f"La predicción es: {prediccion}")


# Cargar el modelo SVM entrenado
modelo_guardado = load('./Models/svm_model.joblib')

# Configurar la interfaz de Tkinter
root = tk.Tk()
root.title("Predicción de Imagen")

# Crear un botón para seleccionar la imagen y predecir su clase
btn_predecir = tk.Button(
    root, text="Seleccionar Imagen y Predecir", command=predecir_imagen)
btn_predecir.pack(pady=20)

# Ejecutar la aplicación
root.mainloop()
