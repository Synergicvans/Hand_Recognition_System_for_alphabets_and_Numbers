# Hand Gesture Recognition System

There are two main files in this project named **Project 7** and **Project 5**, both utilizing the same methodology and components. The key difference is that:

- **Project 7** is designed for recognizing **alphabets** (A-Z).
- **Project 5** is adapted for recognizing **numbers** (0-9).

Both files allow for data collection, training of a gesture recognition model, and performing real-time recognition with the appropriate dataset.

---

## Features

1. **Hand Gesture Data Collection**  
   - Captures 3D hand landmark positions using **Mediapipe Hands**.
   - Saves gesture data (alphabetic or numeric labels) in a CSV file.
   - Provides real-time visualization of hand landmarks while recording data.

2. **Model Training for Gesture Recognition**  
   - Trains a **Random Forest Classifier** on the collected data.
   - Evaluates the model's accuracy and saves it for later use.

3. **Real-Time Gesture Recognition**  
   - Uses the trained model to predict hand gestures in real-time from a webcam stream.
   - Displays both the predicted gesture and hand landmarks on the screen.

---

## Prerequisites

Before starting, ensure the following:

1. Python 3.8 or higher is installed.
2. A functioning webcam is available for data collection and real-time recognition.
3. You have installed the required dependencies (details in [Dependencies](#dependencies)).

---

## Usage Instructions

### **1. Data Collection**
You can collect data separately for alphabets (Project 7) or numbers (Project 5).

#### For Alphabets (Project 7):
Run the `collect_gesture_data.py` script with a focus on **A-Z** gestures:
```shell script
python collect_gesture_data.py
```

#### For Numbers (Project 5):
Run the same `collect_gesture_data.py` script but provide numeric labels (0-9):
```shell script
python collect_gesture_data.py
```

**Steps:**
1. Enter the gesture/label you want to collect data for (e.g., an alphabet or a number) when prompted.
2. Position your hand in front of the webcam.
3. The script will:
   - Detect 21 hand landmarks for your gesture.
   - Save the landmark data (x, y, z coordinates) and the label into a CSV file (`gesture_data.csv`).
4. Press `Q` to exit the data collection process.

---

### **2. Training the Recognition Model**
Once data is collected (for either alphabets or numbers), train a machine learning model using the `train_model.py` script.

```shell script
python train_model.py
```

**Steps:**
1. The script reads hand landmark data from the `gesture_data.csv` file.
2. It trains a **Random Forest Classifier** to associate gestures with their labels (alphabetic or numeric).
3. The accuracy of the model is evaluated and displayed.
4. The trained model is saved as `gesture_model.pkl`.

---

### **3. Real-Time Gesture Recognition**
After training the model, use it to recognize gestures in real-time using `real_time_recognition.py`.

#### For Real-Time Alphabet Recognition (Project 7):
```shell script
python real_time_recognition.py
```

#### For Real-Time Number Recognition (Project 5):
Run the same script, but ensure the collected data and model are for numeric gestures (0-9).

**Steps:**
1. Ensure the trained model `gesture_model.pkl` is in the project directory.
2. Start the script to activate live gesture recognition with your webcam.
3. The system:
   - Detects real-time hand landmarks using Mediapipe.
   - Passes these landmarks to the trained model to predict the corresponding gesture.
   - Displays the predicted gesture along with hand landmarks on the screen.
4. Press `Q` to exit.

---

## Dependencies

Install the dependencies using the following command:
```shell script
pip install -r requirements.txt
```

### Key Libraries:
- **Mediapipe**: For detecting hand landmarks.
- **OpenCV**: For processing webcam video streams.
- **Scikit-learn**: For training and evaluating the Random Forest model.
- **Pandas**: For manipulating and reading the gesture dataset.
- Additional libraries are listed in the `requirements.txt`.

---

## Project Structure

```
├── collect_gesture_data.py       # Script to collect hand gesture data
├── train_model.py                # Script to train the gesture recognition model
├── real_time_recognition.py      # Real-time gesture recognition script
├── gesture_data.csv              # Collected landmarks and gesture labels
├── gesture_model.pkl             # Trained model file
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
├── Project7/                     # Folder with files for alphabet recognition (A-Z)
├── Project5/                     # Folder with files for number recognition (0-9)
```

---

## Examples of Workflow

### **For Project 7 (Alphabet Recognition):**
1. Collect data for a subset or all of the alphabets (e.g., A-Z) using `collect_gesture_data.py`.
2. Train the model for alphabets using `train_model.py`.
3. Recognize alphabets in real-time using `real_time_recognition.py`.

### **For Project 5 (Number Recognition):**
1. Collect numeric gesture data (e.g., 0-9) using `collect_gesture_data.py`.
2. Train the model for numbers using `train_model.py`.
3. Recognize numbers in real-time with `real_time_recognition.py`.

> **Note**: For best results, ensure that gestures are well-defined and consistent during data collection and live recognition.

---

## License

The project is licensed under the **MIT License**. Check the `LICENSE` file for more details.

---

We highly encourage new ideas, feedback, and contributions! Feel free to raise Issues or submit Pull Requests if you’d like to improve this project. 😊
