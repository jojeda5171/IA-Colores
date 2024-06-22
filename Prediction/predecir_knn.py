import os
import cv2
import numpy as np
from joblib import load

# Ruta de la carpeta que contiene las imágenes a predecir
ruta_img = "./Img/rojo.jpg"

# Cargar el modelo KNN entrenado
modelo_guardado = load('./Models/svm_model.joblib')

# Lista para almacenar las predicciones
predicciones = []

# Iterar sobre cada imagen en la carpeta de predicción
# for imagen_nombre in os.listdir(ruta_carpeta_prediccion):
ruta_imagen = os.path.join(ruta_img)
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)  # Lee la imagen a color

# Cambiar tamaño de la imagen y aplicar filtro de mediana (mismo preprocesamiento que en entrenamiento)
imagen = cv2.resize(imagen, (50, 50), interpolation=cv2.INTER_LINEAR)
imagen = cv2.medianBlur(imagen, 3)

# Normalizar los valores de píxeles
imagen = imagen / 255.0

# Flattening para convertir la imagen en un vector
matriz_flattened = imagen.flatten()

# Realizar la predicción utilizando el modelo cargado
prediccion = modelo_guardado.predict([matriz_flattened])[0]
predicciones.append(prediccion)

# Mostrar las predicciones
# for imagen_nombre, prediccion in zip(os.listdir(ruta_carpeta_prediccion), predicciones):
print(f"Predicción: {prediccion}")
