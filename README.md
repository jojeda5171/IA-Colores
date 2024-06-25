# Clasificación de Colores con KNN y SVM

Este proyecto implementa un sistema de clasificación de colores utilizando los algoritmos K-Nearest Neighbors (KNN) y Support Vector Machines (SVM). Adicionalmente, se utiliza el modelo YOLO para la detección de personas en tiempo real.

## Estructura del Proyecto

### Classification
knn.py
svm.py
svm1.py
Data
Colors
amarillo
azul
blanco
naranja
negro
rojo
verde
Models
svm_model.zip
svm_model_INTER_LANCZOS4.zip
Yolo
coco.names
yolov3.cfg
main
main.py


## Descripción de los Archivos

### knn.py
Este script entrena un modelo KNN para la clasificación de colores. Las imágenes se leen, redimensionan, aplican un filtro de mediana, y se normalizan antes de ser aplanadas y utilizadas como datos de entrada para el modelo KNN.

### svm.py
Similar a `knn.py`, este script entrena un modelo SVM con un kernel RBF. Las imágenes pasan por el mismo preprocesamiento que en el script de KNN.

### svm1.py
Este script es una variante de `svm.py` con funcionalidades adicionales, como la impresión del número de vectores de soporte utilizados por el modelo SVM.

### main.py
Este script carga el modelo SVM entrenado y utiliza YOLO para la detección de personas en tiempo real. Para cada persona detectada, clasifica el color del torso y las piernas.

## Dependencias

Para ejecutar estos scripts, necesitas instalar las siguientes bibliotecas:

- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib

Puedes instalar todas las dependencias usando `pip`:

```bash
pip install opencv-python numpy pandas scikit-learn joblib
```

## Entrenamiento de Modelos
### KNN
Para entrenar el modelo KNN, simplemente ejecuta:
```bash
python knn.py
```

### SVM
Para entrenar el modelo SVM, simplemente ejecuta:
```bash
python svm.py
```
O bien:
```bash
python svm1.py
```

## Detección en Tiempo Real
Para ejecutar la detección en tiempo real utilizando YOLO y el modelo SVM entrenado:
```bash
python main.py
```

## Estructura de la Carpeta de Datos
Asegúrate de que las imágenes estén organizadas en carpetas según su color dentro de ./Data/Colors/. Por ejemplo:
```bash
Data
   Colors
      amarillo
         imagen1.jpg
         imagen2.jpg
      azul
         imagen1.jpg
         imagen2.jpg
      ...
```

## Guardado de Modelos
Los modelos entrenados se guardan en la carpeta ./Models/. Asegúrate de que esta carpeta exista antes de ejecutar los scripts.

## Resultados y Logs
El script main.py guarda las detecciones y las clasificaciones de color en un archivo de log llamado detections.log.

## Créditos
Proyecto desarrollado por José Ojeda & Sebastián Palate.

Este archivo README proporciona una guía completa sobre cómo utilizar y entender tu proyecto, facilitando a otros usuarios la ejecución y comprensión del mismo.
