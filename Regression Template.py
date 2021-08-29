# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train_dataset
train_dataset = pd.read_csv('../input/30-days-of-ml/train.csv', index_col="id")
x_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values

# Importing the test_dataset
test_dataset = pd.read_csv('../input/30-days-of-ml/test.csv', index_col="id")
x_test = test_dataset.iloc[:, :].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()

for count in range(0, 10):
    x_train[:, count] = labelencoder_x.fit_transform(x_train[:, count])
    x_test[:, count] = labelencoder_x.fit_transform(x_test[:, count])


# onehotencoder = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(categories='auto'), [A, B, C, D, E, F, G, H, I, J, K])],
#     # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'  # Leave the rest of the columns untouched
# )
# copy_x_train = x_train[:, 1: 11]
# copy_x_train = onehotencoder.fit_transform(copy_x_train)
# print(copy_x_train)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
regressor = RandomForestRegressor(random_state=1, n_estimators=500, n_jobs=-1, warm_start=True)
regressor.fit(x_train, y_train)
rfr_predict = regressor.predict(x_test)

gbr_model = GradientBoostingRegressor(random_state=1, n_estimators=500)
gbr_model.fit(x_train, y_train)
gbr_val_predictions = gbr_model.predict(x_test)

final_prediction = (rfr_predict + gbr_val_predictions) /2

data = pd.read_csv("../input/30-days-of-ml/sample_submission.csv")
data['target'] = final_prediction
data.to_csv('sample_submission_3.csv', index=False)
