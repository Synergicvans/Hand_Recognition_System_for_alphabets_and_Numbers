import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Access the webcam or provide a video file
cap = cv2.VideoCapture(0)  # Use 0 for the first webcam, or replace with file path

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue

    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        print("Hands detected!")
        for hand_landmarks in results.multi_hand_landmarks:
            print(hand_landmarks)  # Print hand landmarks to console
    else:
        print("No hands detected.")

    # Press 'q' to exit the webcam preview
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
