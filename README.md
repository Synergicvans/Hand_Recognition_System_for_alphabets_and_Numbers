# Hand Gesture Recognition System

This project is a comprehensive **Hand Gesture Recognition System** that enables users to collect hand gesture data, train a fingerprinting machine learning model, and achieve real-time recognition of gestures utilizing a webcam. It uses **Mediapipe** for detecting hand landmarks, **scikit-learn** for machine learning, and **OpenCV** for camera integration.

---

## Features

1. **Hand Gesture Data Collection**  
   - Captures hand landmark positions (3D coordinates) using **Mediapipe Hands**.
   - Records data along with a label (e.g., alphabet characters) and saves it in a CSV format.
   - Allows real-time visualization of hand landmark detection.
   
2. **Gesture Model Training**  
   - Trains a **Random Forest Classifier** to recognize gestures using the collected data.
   - Automatically splits data into training and testing sets and displays model accuracy.
   - Saves the trained model for future use.

3. **Real-Time Gesture Recognition**  
   - Uses a pre-trained model to identify hand gestures in real-time from a webcam feed.
   - Displays the predicted gesture on the screen along with a live stream of hand landmarks.
   - Provides instant and intuitive feedback for gestures.

---

## Prerequisites

Make sure you have the following installed/setup before running the project:

1. Python 3.8 or higher.
2. A working webcam for real-time processing.
3. Install project dependencies from the [`requirements.txt`](#dependencies).

---

## Usage Instructions

### 1. Data Collection
Use the `collect_gesture_data.py` script to gather hand gesture data:

```shell script
python collect_gesture_data.py
```

#### Steps:
1. Upon running the script, enter the label for the gesture you wish to record (e.g., a character: `A`, `B`, etc.).
2. Position your hand in front of the webcam, ensuring proper visibility.
3. The script will:
   - Detect 21 hand landmarks using Mediapipe.
   - Log the 3D landmark coordinates (`x`, `y`, `z`) alongside the label into a CSV file named `gesture_data.csv`.
4. Press `Q` to stop data collection at any time.

Collected data can later be used for training the gesture recognition model.

---

### 2. Model Training
Use the `train_model.py` script to train a Random Forest classifier on the collected data:

```shell script
python train_model.py
```

#### Steps:
1. The script reads your dataset from the `gesture_data.csv` file.
2. It splits the data into training and testing sets (default: 80% training, 20% testing).
3. A **Random Forest Classifier** is trained to map hand landmarks to their corresponding gestures.
4. The script evaluates the model and outputs the accuracy of predictions on the test set.
5. The trained model is saved as `gesture_model.pkl` for use in real-time recognition.

- Make sure `gesture_data.csv` is populated before running this script.

---

### 3. Real-Time Gesture Recognition
Use the `real_time_recognition.py` script to perform real-time gesture recognition:

```shell script
python real_time_recognition.py
```

#### Steps:
1. Ensure that the trained model `gesture_model.pkl` is available in the project directory.
2. Run the script to start webcam-based real-time recognition.
3. The system will:
   - Detect hand landmarks in live webcam feed using **Mediapipe**.
   - Pass the landmarks to the trained machine learning model to predict the gesture.
   - Display predictions and corresponding hand landmarks in the webcam feed.
4. Press `Q` to exit the real-time recognition.

---

## Dependencies

To set up the environment, install the required libraries with the following command:

```shell script
pip install -r requirements.txt
```

### Requirements Summary
- **Mediapipe**: For hand landmark detection.
- **OpenCV**: For webcam-based input and visual output.
- **scikit-learn**: For training and evaluating the Random Forest model.
- **Pandas**: To process gesture data (CSV file).
- Other supporting libraries are specified in `requirements.txt`.

---

## Project Structure

```
├── collect_gesture_data.py       # Script for collecting gesture data
├── train_model.py                # Script for training the gesture recognition model
├── real_time_recognition.py      # Script for gesture recognition in real-time
├── gesture_data.csv              # Collected gesture data (output of data collection)
├── gesture_model.pkl             # Saved trained model (output of model training)
├── requirements.txt              # List of project dependencies
└── README.md                     # Project documentation
```

---

## Example Workflow

1. Run `collect_gesture_data.py` to gather data for gestures (e.g., `A`, `B`, or custom gestures). The labels and their associated hand landmarks will be saved into `gesture_data.csv`.
   
2. Train the model using `train_model.py`. This creates the model file `gesture_model.pkl`.

3. Finally, use `real_time_recognition.py` to predict hand gestures in a live video feed.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for detailed information.

---

We encourage contributions to the project. Whether you want to suggest an improvement, report a bug, or implement a feature, feel free to create an issue or submit a pull request! 😊
