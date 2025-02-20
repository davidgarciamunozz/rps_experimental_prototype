import cv2
import mediapipe as mp
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("modelo_ppt.pkl")

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediccion = "Desconocido"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas de los puntos de la mano
            datos = []
            for lm in hand_landmarks.landmark:
                datos.extend([lm.x, lm.y, lm.z])

            # Hacer predicción con el modelo si hay suficientes datos
            if len(datos) == 63:  # 21 puntos * 3 coordenadas (x, y, z)
                prediccion = modelo.predict([datos])[0]

    # Mostrar predicción en pantalla
    cv2.putText(frame, f"Gesto Detectado: {prediccion}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detección en Tiempo Real", frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
