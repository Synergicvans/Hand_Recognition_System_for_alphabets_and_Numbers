import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("gesture_data.csv")

# Separate features (landmarks) and labels (characters)
X = data.iloc[:, :-1]  # Landmark points
y = data.iloc[:, -1]  # Labels (characters)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model to disk
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(clf, f)
    print("Model saved as gesture_model.pkl")
