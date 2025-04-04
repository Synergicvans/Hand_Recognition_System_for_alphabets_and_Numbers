import cv2
import pickle
import mediapipe as mp
import numpy as np

# Load model
MODEL_FILE = './model.p'
model_dict = pickle.load(open(MODEL_FILE, 'rb'))
model = model_dict['model']

# Labels
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Frame not captured.")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize
            x_min, y_min = min(x_), min(y_)
            data_aux = [(landmark.x - x_min) for landmark in hand_landmarks.landmark]
            data_aux += [(landmark.y - y_min) for landmark in hand_landmarks.landmark]

            # Predict
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and label
            x_min_px, y_min_px = int(x_min * W), int(y_min * H)
            x_max_px, y_max_px = int(max(x_) * W), int(max(y_) * H)
            cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), (255, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x_min_px, y_min_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
