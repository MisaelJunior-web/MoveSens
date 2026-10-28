import cv2 
import mediapipe as mp
import csv
from pathlib import Path

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a webcam")
    raise SystemExit

gesto = input("Digite o gesto que deseja capturar (ex: 'coracao', 'rock'): ").strip().lower()

gestos_validos = { "coracao", "rock", "joia", "hangloose", "negativo", "paz"}

if gesto not in gestos_validos:
    print(f"Gesto inválido. Gestos válidos: {gestos_validos}")
    raise SystemExit

pasta = Path("dados")
pasta.mkdir(exist_ok=True)
caminho_csv = pasta / f"{gesto}.csv"

contador = 0

with open(caminho_csv, "a", newline="") as f, mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    writer = csv.writer(f)

    while True:
        success, img = cap.read()

        if not success:
            print("Não foi possível ler a câmera.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                row=[]

                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                writer.writerow(row)
                contador += 1 

        cv2.putText(
            img,
            f"Gesto: {gesto} | Coletados: {contador}",
            (10,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.imshow("Coleta de Dados", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        
cap.release()
cv2.destroyAllWindows()

print(f"Coleta de dados finalizada: {contador} novos registros.")
                    