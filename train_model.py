import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

path_in = 'dados/melb_data.csv'
data = pd.read_csv(path_in)

data_process = data.dropna(axis=0)

y = data_process.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data_process[melbourne_features]
X.describe()
X.head()

melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
print("Actual values")
print(y.head().to_list())

# Validando modelo
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
