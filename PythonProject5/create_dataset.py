import os
import pickle
import cv2
import mediapipe as mp
from tqdm import tqdm

# Mediapipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
OUTPUT_FILE = 'data.pickle'

# Initialize data lists
data, labels = [], []

# Process each class directory
print("Processing dataset...")
for class_name in tqdm(os.listdir(DATA_DIR), desc="Classes"):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip invalid directories

    for img_file in tqdm(os.listdir(class_dir), desc=f"Class {class_name} images", leave=False):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract normalized hand landmark coordinates
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                # Normalize coordinates by subtracting minimum values
                data_aux = []
                x_min, y_min = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - x_min)
                    data_aux.append(landmark.y - y_min)

                # Add processed data and label
                data.append(data_aux)
                labels.append(class_name)

# Save processed data
print(f"Saving processed data to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset processing complete.")
