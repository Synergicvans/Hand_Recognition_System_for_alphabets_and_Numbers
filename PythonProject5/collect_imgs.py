import os
import cv2
import argparse
from tqdm import tqdm

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description="Collect hand gesture images for the sign language dataset.")
parser.add_argument('--data_dir', type=str, default='./data', help="Directory to store the dataset.")
parser.add_argument('--num_classes', type=int, default=3, help="Number of gesture classes.")
parser.add_argument('--dataset_size', type=int, default=100, help="Number of images per class.")
parser.add_argument('--camera_index', type=int, default=0, help="Camera index (default: 0).")
args = parser.parse_args()

DATA_DIR = args.data_dir
number_of_classes = args.num_classes
dataset_size = args.dataset_size
camera_index = args.camera_index

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not initialize the camera. Try a different `--camera_index` value.")
    exit(1)

# Loop through all classes to collect images
for class_id in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    os.makedirs(class_dir, exist_ok=True)  # Create directory for each class

    print(f"\nCollecting data for class {class_id}...")

    # Wait for user to get ready
    print('Press "q" to start collecting images.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue

        cv2.putText(frame, "Get ready! Press 'q' to start!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture specified number of images for the class
    print(f"Collecting {dataset_size} images for class {class_id}.")
    for img_count in tqdm(range(dataset_size), desc=f"Class {class_id}"):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Skipping frame due to capture error.")
            continue

        # Optional: Draw class and count info on the frame
        text = f"Class: {class_id}, Count: {img_count + 1}/{dataset_size}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame and save image
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{img_count}.jpg'), frame)

        # Allow a small delay for camera stabilization
        cv2.waitKey(1)

print(f"\nDataset collection complete! Saved in '{DATA_DIR}'.")

# Clean up
cap.release()
cv2.destroyAllWindows()
