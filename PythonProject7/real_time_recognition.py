import mediapipe as mp
import cv2
import pickle

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Start webcam for real-time prediction
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks += [lm.x, lm.y, lm.z]

            # Predict gesture/character
            prediction = model.predict([landmarks])
            predicted_character = prediction[0]
            print(f"Predicted Character: {predicted_character}")

            # Draw landmarks on the screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the prediction on the frame
            cv2.putText(frame, f"Character: {predicted_character}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, cv2.LINE_AA)

    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
