import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar los datos
df = pd.read_csv("data_ppt.csv")

# Separar características (X) y etiquetas (y)
X = df.drop(columns=["clase"])  
y = df["clase"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo
accuracy = modelo.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
joblib.dump(modelo, "modelo_ppt.pkl")
