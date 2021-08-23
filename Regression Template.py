# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train_dataset
train_dataset = pd.read_csv('train.csv', index_col="id")
x_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values

# Importing the test_dataset
test_dataset = pd.read_csv('test.csv', index_col="id")
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


# dividing training and validating data
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_predict = regressor.predict(x_test)

"""# to find the error
from sklearn import metrics
err = metrics.mean_squared_error(y_val, y_predict)
err = np.square(err)
print(err)"""

data = pd.read_csv("sample_submission.csv")
data['target'] = regressor.predict(x_test)
data.to_csv('sample_submission.csv')