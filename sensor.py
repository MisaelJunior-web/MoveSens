import cv2
import mediapipe as mp
import joblib
from pathlib import Path

model = joblib.load("modelo_gestos.pkl")

labels = [
    "joia",
    "paz",
    "negativo",
    "hangloose",
    "rock",
    "coracao"
]

pasta_imagens = Path("imagens")

imagens = {
    nome: cv2.imread(str(pasta_imagens / f"{nome}.jpg"))
    for nome in labels
}

for nome, img in imagens.items():
    if img is None:
        print(f"Aviso: imagem {nome}.jpg não encontrada.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a webcam.")
    raise SystemExit

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, img = cap.read()

            if not success:
                print("Não foi possível ler a câmera.")
                break

            img_rgb = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB
            )

            results = hands.process(img_rgb)

            gesto_detectado = None
            confianca = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []

                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                pred = model.predict([row])[0]
                probabilidades = model.predict_proba([row])[0]

                indice_classe = list(model.classes_).index(pred)
                confianca = float(probabilidades[indice_classe])

                indice = int(pred)

                if 0 <= indice < len(labels):
                    gesto_detectado = labels[indice]

            if gesto_detectado is not None:
                texto = (
                    f"{gesto_detectado} "
                    f"({confianca * 100:.1f}%)"
                )

                cv2.putText(
                    img,
                    texto,
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                img_gesto = imagens.get(gesto_detectado)

                if img_gesto is not None:
                    img_gesto = cv2.resize(
                        img_gesto, (180, 180)
                    )

                    altura, largura = img.shape[:2]

                    if altura >= 180 and largura >= 180:
                        img[0:180, 0:180] = img_gesto

            else:
                cv2.putText(
                    img,
                    "Nenhuma mao detectada",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("MoveSens - Detector de Gestos", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
