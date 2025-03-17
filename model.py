import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle

# Load data
df = pd.read_csv("mail_data.csv")

# Replace NaN values with empty strings
data = df.where(pd.notnull(df), "")

# Encode labels ('spam' = 0, 'ham' = 1)
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Split into input and target
x = data['Message']
y = data['Category'].astype(int)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = vectorizer.fit_transform(x_train)
x_test_features = vectorizer.transform(x_test)   # ✅ Fixed (transform instead of fit_transform)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Train model
model = RandomForestClassifier()
model.fit(x_train_features, y_train)  # ✅ Fixed (removed double fitting)

# Evaluate on training data
train_predictions = model.predict(x_train_features)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Evaluate on test data
test_predictions = model.predict(x_test_features)
test_accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

print("\n--- Test Results ---")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, test_predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_predictions))

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load and test the model with sample input
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Sample input
sample_input = ["Nah I don't think he goes to usf, he lives around here though"]

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

input_data = vectorizer.transform(sample_input)
prediction = loaded_model.predict(input_data)

result = "Ham" if prediction[0] == 1 else "Spam"
print("\nSample Prediction:", result)
