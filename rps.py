import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Función para clasificar jugada
def clasificar_jugada(hand_landmarks):
    dedos_abiertos = [False] * 5  # Pulgar a meñique
    tip_ids = [4, 8, 12, 16, 20]  # Índices de las puntas de los dedos en MediaPipe

    for i in range(5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            dedos_abiertos[i] = True

    if all(dedos_abiertos):  
        return "Papel 🖐️"
    elif not any(dedos_abiertos):
        return "Piedra ✊"
    elif dedos_abiertos[1] and dedos_abiertos[2] and not dedos_abiertos[0] and not dedos_abiertos[3] and not dedos_abiertos[4]:  
        return "Tijeras ✌️"
    else:
        return "No reconocido ❓"

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            jugada = clasificar_jugada(hand_landmarks)
            cv2.putText(frame, f"Jugada: {jugada}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("rps", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
