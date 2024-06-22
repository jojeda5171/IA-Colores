# obtener una observacion del dataset y reconstruir la imagen original
import cv2
import numpy as np
import pandas as pd
import math

# Leer el archivo CSV con el dataset
df = pd.read_csv('./Data/dataset_colors.csv')

# Obtener una observación del dataset
observacion = df[0:1]

# Obtener la etiqueta de la observación
etiqueta = observacion['etiqueta'].values[0]

# Obtener la matriz de la observación
matriz_binaria = observacion.drop('etiqueta', axis=1).values[0]

# desnormalizar los valores de píxeles
matriz_binaria = math.abs(matriz_binaria * 255.0)

# Reconstruir la imagen original
imagen = np.reshape(matriz_binaria, (230, 230)).astype(np.uint8)

# Mostrar la imagen
cv2.imshow(etiqueta, imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
