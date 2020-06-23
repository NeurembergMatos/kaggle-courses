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


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


for max_leaf_nodes in [5, 50, 100, 250, 500, 1000, 1750, 2500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# Modelo final
model_final = DecisionTreeRegressor(max_leaf_nodes=1000)
model_final.fit(X, y)
