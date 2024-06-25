import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

# Ruta a la carpeta de datos
ruta_carpeta = "./Data/Colors"

datos = []
etiquetas = []

# Recorrer cada carpeta dentro de la carpeta de datos
for carpeta in os.listdir(ruta_carpeta):
    # Leer la ruta de la carpeta a través de la carpeta actual
    ruta_carpeta_actual = os.path.join(ruta_carpeta, carpeta)

    # Asignar la etiqueta de la carpeta a la variable etiqueta
    etiqueta = carpeta

    # Recorrer cada imagen en la carpeta actual
    for imagen_nombre in os.listdir(ruta_carpeta_actual):
        # Leer la ruta de la imagen a través de la carpeta actual
        ruta_imagen = os.path.join(ruta_carpeta_actual, imagen_nombre)

        # Leer la imagen a color
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

        # Cambiar tamaño de la imagen y aplicar filtro de mediana
        imagen = cv2.resize(imagen, (50, 50), interpolation=cv2.INTER_LANCZOS4)

        # Aplicar filtro de mediana
        imagen = cv2.medianBlur(imagen, 3)

        # Normalizar los valores de píxeles
        imagen = imagen / 255.0

        # Aplanar la imagen y agregar a los datos
        matriz_flattened = imagen.flatten()
        datos.append(matriz_flattened)
        etiquetas.append(etiqueta)

datos = np.array(datos)
etiquetas = np.array(etiquetas)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    datos, etiquetas, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo SVM: {accuracy}')

# Obtener los índices de los vectores de soporte
vectores_de_soporte_indices = svm.support_

# Contar el número de vectores de soporte
num_vectores_de_soporte = len(vectores_de_soporte_indices)

print(f'Número de vectores de soporte: {num_vectores_de_soporte}')

# Guardar el modelo si es necesario
dump(svm, './Models/svm_model_INTER_LANCZOS4.joblib')

# Guardar el DataFrame si es necesario
""" df = pd.DataFrame(datos)
df['etiqueta'] = etiquetas
df.to_csv('./Data/dataset_colors.csv', index=False) """
