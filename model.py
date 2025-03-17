pip install scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("/content/drive/MyDrive/mail_data.csv")
print(df)

data = df.where(pd.notnull(df), " ")

data.head()

data.info()

data.shape

data.loc[data['Category'] == 'spam','Category',] = 0
data.loc[data['Category'] == 'ham','Category',] = 1

x = data['Message']
print(x)

y = data['Category']
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

print(x.shape)
print(x_train.shape)
print(x_test.shape)

print(y.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.fit_transform(x_test)
x_train_features.shape[1]

with open("vectorizer.pickle", "wb") as f:
    pickle.dump(feature_extraction,f)

y_train = y_train.astype('int')
y_test = y_test.astype("int")

print(x_train)

print(x_train_features)

model = RandomForestClassifier()

model.fit(x_train_features, y_train)

model.fit(x_test_features, y_test)

prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

print("Accuracy on training data : ", accuracy_on_training_data)

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

print("Accuracy on test data : ", accuracy_on_test_data)

accuracy = accuracy_score(y_test, prediction_on_test_data)
precision = precision_score(y_test, prediction_on_test_data)
recall = recall_score(y_test, prediction_on_test_data)
f1 = f1_score(y_test, prediction_on_test_data)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Print a classification report for more detailed evaluation
print("\nClassification Report:\n", classification_report(y_test, prediction_on_test_data))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, prediction_on_test_data))

input = [""]

input_data = feature_extraction.transform(input)

prediction = model.predict(input_data)

print(prediction)

if (prediction[0]==1):
  print("Ham mail")
else:
  print("Spam mail")

with open("model_pickle","wb") as f:
  pickle.dump(model,f)

with open("model_pickle","rb") as f:
  mp = pickle.load(f)

input = ["Nah I don't think he goes to usf, he lives around here though"]

input_data = feature_extraction.transform(input)

mp.predict(input_data)
