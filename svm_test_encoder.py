
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
from sklearn.model_selection import GridSearchCV



data = pd.read_csv('data_classification_training.csv')


data.dropna()
data['vpa'] = data['vpa'].str.lower()
data['weekday'] = pd.to_datetime(data['duration']).dt.weekday
data['hour'] = pd.to_datetime(data['duration']).dt.hour


# Label encoding for categorical feature (label,vpa)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
target = data['label']

vpa_encoder = LabelEncoder()
data['vpa_1'] = vpa_encoder.fit_transform(data['vpa'])

# Extract target variable
features = data[['amount','hour','weekday','vpa_1']]

print(features.shape)
print(target.shape)
# print(target)
# print(features)


#seprating training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Feature scaling for numerical features (amount and timestamp)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['amount', 'weekday','hour','vpa_1']])
X_test_scaled = scaler.transform(X_test[['amount', 'weekday','hour','vpa_1']])

print("Shape of X_train_scaled:", X_train_scaled.shape)
# print("Shape of X_train_vpa_vectorized:", X_train_vpa_vectorized.shape)
print("Shape of X_test_scaled:", X_test_scaled.shape)
# print("Shape of X_test_vpa_vectorized:", X_test_vpa_vectorized.shape)
# print("Shape of X_train_vpa_vectorized:", pd.DataFrame(X_train_vpa_vectorized.toarray()))



    # ## Experimenting with the different configuration 
    # # Define candidate kernels (uncomment desired options)
    # kernels = {'linear': ('linear', None), 'rbf': ('rbf', SVC(kernel='rbf'))}

    # # Define hyperparameter ranges for grid search (adjust as needed)
    # param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}

    # # Create a grid search object
    # grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='f1_macro')

    # # Fit the grid search to your data (replace with your features)
    # grid_search.fit(X_train_scaled, y_train)

    # # Best performing kernel and hyperparameters
    # best_kernel = grid_search.best_estimator_.kernel
    # best_C = grid_search.best_params_['C']
    # best_gamma = grid_search.best_params_['gamma']

    # # Create an SVM model with the best configuration

    # svm_clf = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma)


svm_clf = SVC(kernel='rbf', C=10.6)  # Experiment with different kernels (e.g., 'rbf') and C values
# svm_clf = SVC(kernel='rbf', C=2.0, gamma=3.5)

svm_clf.fit(X_train_scaled, y_train)

# Assuming new data follows the same format as features

new_data = pd.read_csv('predict.csv')


# Pre-process new data using the fitted pipeline
new_data['weekday'] = pd.to_datetime(new_data['duration']).dt.weekday
new_data['hour'] = pd.to_datetime(new_data['duration']).dt.hour
scaler = StandardScaler()
vectorizer = TfidfVectorizer(max_features=500)

vpa_encoder_1 = LabelEncoder()
new_data['vpa_1'] = vpa_encoder_1.fit_transform(new_data['vpa'])
 # Extract target variable

new_data_scaled = scaler.fit_transform(new_data[['amount', 'weekday','hour','vpa_1']])

# new_data_combined = np.hstack((new_data_scaled, new_data_vectorized.toarray()))  # Or use np.vstack or concatenation with axis=1

print(new_data_scaled.shape)

# X_test_combined = X_test_combined.reshape(1,-1)
predictions = svm_clf.predict(new_data_scaled)

# Decode predictions using the label encoder for interpretation (optional)
predicted_labels = label_encoder.inverse_transform(predictions)
print("Predicted labels:", predicted_labels)

actual_labels = label_encoder.inverse_transform(y_test)

# print("actual labels:", actual_labels)

