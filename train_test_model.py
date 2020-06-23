
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Carregando dados
path_in = 'dados/melb_data.csv'
data = pd.read_csv(path_in)

# Seleção de variáveis e processamento
target = ['Price']
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
data = data[melbourne_features + target]
data.dropna(axis=0, inplace=True)

y = data[target[0]]
X = data[melbourne_features]
X.describe()
X.head()

# Separando base de dados em treino e validação
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Treinando modelo
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)

# Validando modelo fora da base de treinamento
val_predictions = melbourne_model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)
print("MAE do modelo:", mae)

# Testando overfitting

