import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

# --- 1. Datos y Entrenamiento del Modelo ---

print("Entrenando modelo de ML (KNN)...")

# 'X' son nuestros datos (features). 
# Son los valores [Blue, Green, Red] de los colores.
# (OpenCV usa BGR por defecto)
X_data = [
    # Ejemplos de ROJO
    [0, 0, 200], [0, 0, 255], [30, 20, 220], [15, 15, 180],
    
    # Ejemplos de VERDE
    [0, 200, 0], [0, 255, 0], [20, 220, 30], [15, 180, 15],
    
    # Ejemplos de AZUL
    [200, 0, 0], [255, 0, 0], [220, 30, 20], [180, 15, 15],
    
    # Ejemplos de AMARILLO
    [0, 200, 200], [0, 255, 255], [30, 220, 230], [15, 180, 180],
    
    # Ejemplos de "OTROS" (para reducir ruido)
    [0, 0, 0], [255, 255, 255], [128, 128, 128] 
]

# 'y' son nuestras etiquetas (labels).
y_data = [
    'Rojo', 'Rojo', 'Rojo', 'Rojo',
    'Verde', 'Verde', 'Verde', 'Verde',
    'Azul', 'Azul', 'Azul', 'Azul',
    'Amarillo', 'Amarillo', 'Amarillo', 'Amarillo',
    'Otro', 'Otro', 'Otro'
]

# Convertimos las listas a arrays de NumPy
# Especificamos el tipo de dato como 'uint8' (entero 0-255)
X = np.array(X_data, dtype=np.uint8)
y = np.array(y_data)

# Inicializamos el clasificador KNN
model = KNeighborsClassifier(n_neighbors=3)

# ¡Entrenamos el modelo!
model.fit(X, y)

print("¡Modelo entrenado!")

# --- 2. Configuración de la Cámara ---

# ARREGLO 1: Cambiado a '1' (como tu primer script)
# Si '1' no funciona, prueba con '0' o '2'
print("Iniciando camara...")
cap = cv2.VideoCapture(1) 
time.sleep(1) # Pequeña pausa para que inicie la cámara

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Revisa el índice (0, 1, ...)")
    exit()

# Definimos el tamaño de nuestra Región de Interés (ROI)
roi_size = 100
    
print("Presiona 'q' para salir.")

# --- 3. Bucle Principal de Detección ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame.")
        break

    # Voltear el frame (efecto espejo)
    frame = cv2.flip(frame, 1)

    # Obtenemos las dimensiones del frame
    (h, w) = frame.shape[:2]

    # Calculamos las coordenadas del cuadro (ROI) en el centro
    x1 = (w - roi_size) // 2
    y1 = (h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # 4. Extraer la ROI y calcular el color promedio
    
    # Dibujamos el rectángulo en el frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Extraemos la región
    roi = frame[y1:y2, x1:x2]
    
    # Calculamos el color promedio (B, G, R) de esa región
    avg_color_tuple = cv2.mean(roi)[:3]
    
    # ARREGLO 2: Convertir el promedio a un formato de datos consistente
    # cv2.mean() devuelve floats, pero el modelo se entrenó con enteros (uint8).
    # Debemos asegurarnos de que el tipo de dato sea el mismo.
    avg_color = np.array(avg_color_tuple, dtype=np.uint8)
    
    # Lo convertimos a un formato que el modelo pueda predecir
    # (Necesita ser un array 2D, por eso los dobles corchetes [[...]])
    # El modelo espera (n_samples, n_features), así que le damos (1, 3)
    color_para_predecir = np.array([avg_color])

    # 5. Predecir el color con el modelo de ML
    prediccion = model.predict(color_para_predecir)
    nombre_color = prediccion[0] # El resultado es una lista, tomamos el primer item

    # 6. Mostrar el resultado
    
    # Ponemos el texto en la pantalla
    texto_resultado = f"Color detectado: {nombre_color}"
    cv2.putText(frame, texto_resultado, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostramos un pequeño cuadro con el color promedio detectado
    # Creamos un color BGR (como entero) para que cv2.rectangle lo acepte
    color_cuadro = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
    cv2.rectangle(frame, (10, 10), (60, 60), color_cuadro, -1)
    cv2.putText(frame, "Promedio", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostramos la ventana
    cv2.imshow('Detector de Colores con ML (KNN)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Liberar Recursos ---
cap.release()
cv2.destroyAllWindows()