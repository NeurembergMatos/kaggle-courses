import pandas as pd

path_in = 'dados/melb_data.csv'
data = pd.read_csv(path_in)


def inspect_data(data: pd.DataFrame) -> pd.DataFrame:
    inspection = {
        'colunas': data.columns,
        'tipos': data.dtypes,
        'nulos': data.isna().sum() / data.shape[0]
    }

    inspection = pd.DataFrame(inspection)
    inspection.reset_index(drop=True, inplace=True)

    return inspection


resumo = inspect_data(data)
descricao = data.describe()
descricao[['YearBuilt', 'Date']]

data['Date'].sort_values(ascending=False)
data['YearBuilt'].max()
