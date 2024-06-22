import cv2
import numpy as np
import os
from joblib import load
import logging

# Configuración de logging
logging.basicConfig(filename='detections.log',
                    level=logging.INFO, format='%(asctime)s - %(message)s')

# Cargar el modelo KNN entrenado
model_path = './Models/svm_model.joblib'
knn_model = load(model_path)

# Cargar YOLO
net = cv2.dnn.readNet("./Yolo/yolov3.weights", "./Yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Leer las etiquetas de COCO
with open("./Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configuración de la cámara
cap = cv2.VideoCapture(1)  # 0 para la cámara por defecto

# Crear carpeta para guardar imágenes
if not os.path.exists('detected_parts'):
    os.makedirs('detected_parts')

# Función para procesar y predecir el color


def procesar_y_predecir(imagen, nombre_archivo):
    if imagen.size != 0:
        cv2.imwrite(nombre_archivo, imagen)
        imagen_resized = cv2.resize(imagen, (50, 50))
        imagen_blurred = cv2.medianBlur(imagen_resized, 3)
        imagen_normalized = imagen_blurred / 255.0
        imagen_flattened = imagen_normalized.flatten().reshape(1, -1)
        color_pred = knn_model.predict(imagen_flattened)
        return color_pred[0]
    return "Indefinido"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            # Ajustar las proporciones del torso y las piernas
            torso_y1 = y
            torso_y2 = y + int(h * 0.45)
            piernas_y1 = y + int(h * 0.55)
            piernas_y2 = y + h

            torso = frame[torso_y1:torso_y2, x:x+w]
            piernas = frame[piernas_y1:piernas_y2, x:x+w]

            # Guardar las imágenes del torso y las piernas
            torso_file = f'detected_parts6/torso_{i}.png'
            piernas_file = f'detected_parts6/piernas_{i}.png'

            # Clasificación del color del torso
            color_torso = procesar_y_predecir(torso, torso_file)

            # Clasificación del color de las piernas
            color_piernas = procesar_y_predecir(piernas, piernas_file)

            # Registrar detección en el log
            logging.info(f'Persona {i}: Torso: {
                         color_torso}, Piernas: {color_piernas}')

            # Imprimir colores en la consola
            print(f'Persona {i}: Torso: {
                  color_torso}, Piernas: {color_piernas}')

            # Dibujar el rectángulo y la etiqueta del color en la imagen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'TORSO: {color_torso}', (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'PIERNAS: {color_piernas}', (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame con las detecciones
    cv2.imshow('Person Detection and Color Classification', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
