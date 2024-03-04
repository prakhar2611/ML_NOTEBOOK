import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np

"""# loading the data from csv"""

# Replace 'your_data.csv' with the actual path to your CSV file
data = pd.read_csv('test_cl.csv')
data.dropna()
data['vpa'] = data['vpa'].str.lower()

data['weekday'] = pd.to_datetime(data['duration']).dt.weekday
data['hour'] = pd.to_datetime(data['duration']).dt.hour

# Extract features based on data types

# Label encoding for categorical feature (label)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
# data['vpa_1'] = label_encoder.fit_transform(data['vpa'])
target = data['label']  # Extract target variable

features = data[['amount','vpa','hour','weekday']]

print(features.shape)
print(target.shape)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)



# Feature scaling for numerical features (amount and timestamp)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train[['amount', 'weekday','hour','vpa_1']])
# X_test_scaled = scaler.transform(X_test[['amount', 'weekday','hour','vpa_1']])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['amount', 'weekday','hour']])
X_test_scaled = scaler.transform(X_test[['amount', 'weekday','hour']])

# TF-IDF vectorization for text feature (vpa)
vectorizer = TfidfVectorizer(max_features=500)  # Adjust max_features as needed
X_train_vpa_vectorized = vectorizer.fit_transform(X_train['vpa'])
X_test_vpa_vectorized = vectorizer.transform(X_test['vpa'])

print("Shape of X_train_scaled:", X_train_scaled.shape)
# print("Shape of X_train_vpa_vectorized:", X_train_vpa_vectorized.shape)
print("Shape of X_test_scaled:", X_test_scaled.shape)
# print("Shape of X_test_vpa_vectorized:", X_test_vpa_vectorized.shape)
# print("Shape of X_train_vpa_vectorized:", pd.DataFrame(X_train_vpa_vectorized.toarray()))

X_train_combined = np.hstack((X_train_scaled, X_train_vpa_vectorized.toarray()))  # Or use np.vstack or concatenation with axis=1
X_test_combined = np.hstack((X_test_scaled, X_test_vpa_vectorized.toarray()))


# X_train_combined = np.hstack((X_train_scaled))  # Or use np.vstack or concatenation with axis=1
# X_test_combined = np.hstack((X_test_scaled))

print(X_test_combined.shape)
print(X_train_combined.shape)

# svm_clf = SVC(kernel='linear', C=1.0)  # Experiment with different kernels (e.g., 'rbf') and C values
# svm_clf.fit(X_train_scaled, y_train)

svm_clf = SVC(kernel='linear', C=1.0)  # Experiment with different kernels (e.g., 'rbf') and C values
svm_clf.fit(X_train_combined, y_train)



# Assuming new data follows the same format as features

new_data = pd.read_csv('predict.csv')


# Pre-process new data using the fitted pipeline
new_data['weekday'] = pd.to_datetime(new_data['duration']).dt.weekday
new_data['hour'] = pd.to_datetime(new_data['duration']).dt.hour
scaler = StandardScaler()
vectorizer = TfidfVectorizer(max_features=500)

new_data_scaled = scaler.fit_transform(new_data[['amount', 'weekday','hour']])
new_data_vectorized = vectorizer.fit_transform(new_data['vpa'])

new_data_combined = np.hstack((new_data_scaled, new_data_vectorized.toarray()))  # Or use np.vstack or concatenation with axis=1

print(new_data_combined.shape)
print(X_test_combined.shape)

# X_test_combined = X_test_combined.reshape(1,-1)
# predictions = svm_clf.predict(X_test_scaled)
predictions = svm_clf.predict(X_test_combined)

# Decode predictions using the label encoder for interpretation (optional)
predicted_labels = label_encoder.inverse_transform(predictions)
print("Predicted labels:", predicted_labels)

actual_labels = label_encoder.inverse_transform(y_test)

print("actual labels:", actual_labels)

"""## Using label encode for vpa
### instead of using the vectorizer for the string categroical value

"""
