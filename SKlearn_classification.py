# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FASC_wREBDY7FimvwywO8yCgKiJ86Sh6
"""

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

dataframe = pd.read_csv('data_classification_training.csv')
dataframe.head()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



data = pd.read_csv("data_classification_training.csv")

data['text'] = data['vpa'].str.lower()  # Lowercase text
data['text'] = data['text'].str.replace("[^a-zA-Z0-9\s]", "", regex=True)  # Remove punctuation

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=30)  # Adjust max_features as needed
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression(multi_class='ovr', solver='lbfgs')  # Multi-class classification
model.fit(X_train_features, y_train)

y_pred = model.predict(X_test_features)
print(X_test_features,y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_text = "pay9972233341@paytm"
new_text_features = vectorizer.transform([new_text])
predicted_label = model.predict(new_text_features)[0]
print("Predicted Label:", predicted_label)
