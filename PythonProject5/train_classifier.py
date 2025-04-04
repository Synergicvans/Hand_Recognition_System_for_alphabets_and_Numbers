import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load processed data
DATA_FILE = './data.pickle'
MODEL_FILE = 'model.p'

print(f"Loading data from {DATA_FILE}...")
data_dict = pickle.load(open(DATA_FILE, 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest Classifier
print("Training classifier...")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
print(f"Saving model to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'model': model}, f)

print("Training complete.")
