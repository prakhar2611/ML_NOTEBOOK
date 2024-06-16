
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('data_classification_training.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()

label = 'label'
print(f"Unique classes: {list(train_data[label].unique())}")

predictor = TabularPredictor(label=label).fit(train_data)

test_data = TabularDataset('test.csv')
test_data

y_pred = predictor.predict(test_data)
test_data["label"] = y_pred  # Predictions
test_data

y_pred_proba = predictor.predict_proba(test_data)
y_pred_proba.head()  # Prediction Probabilities

predictor.evaluate(test_data)

predictor.leaderboard(test_data)