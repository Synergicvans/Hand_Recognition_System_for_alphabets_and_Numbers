import mediapipe as mp
import cv2
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create a CSV file to store the gesture data
output_file = "gesture_data.csv"
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row for 21 landmarks (x, y, z) + character label
        header = [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']]
        header.append("label")
        writer.writerow(header)

# Webcam setup
cap = cv2.VideoCapture(0)
character = input("Enter the character (A-Z) for this gesture: ").upper()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmark points
            row = []
            for landmark in hand_landmarks.landmark:
                row += [landmark.x, landmark.y, landmark.z]

            # Include the character label
            row.append(character)

            # Save data to CSV
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Data collected for character {character}")

    # Display the webcam feed
    cv2.imshow('Collecting Data', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
