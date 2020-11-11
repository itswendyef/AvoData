import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump, load

# Cargar datos

df = pd.read_csv("c:/Users/E1_gi/OneDrive/Escritorio/valoresAguacate.csv")

# Eliminar los datos inutiles
f_dataframe = pd.get_dummies(df, columns=['dia', 'mes','ano'])

# Eliminamos el dato que queremos predecir
del f_dataframe['agua']

# creamos la matriz de resultados para entrenar con la matriz de propiedades
X = f_dataframe.to_numpy()
y = df['agua'].to_numpy()


# Dividimos train set para que sea del 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Modelo de regresión


model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=3,
    max_features=0.9,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Guardamos modelo
dump(model, 'sup_learn_avo.joblib')

