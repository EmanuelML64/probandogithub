import pandas as pd
import numpy as np

np.random.seed(42)

jugadores = ["Betts", "Judge", "Soto", "Ohtani", "Alonso",
             "Bellinger", "Rendon", "Ramirez", "Springer", "Acuna"]
pitchers = ["Cole", "Burnes", "Snell", "Scherzer", "Kershaw",
            "Bieber", "deGrom", "Greinke", "Verlander", "Darvish"]

# Creamos una lista vacía
rows = []

# Generamos 30 días × 10 jugadores
for dia in range(1, 31):
    for i in range(10):
        rows.append({
            "Día": dia,
            "Bateador": jugadores[i],
            "Pitcher": np.random.choice(pitchers),
            "AVG_vs_P": round(np.random.uniform(0.200, 0.350), 3),
            "HR_vs_P": np.random.randint(0, 5),
            "Handedness": np.random.choice(["R", "L"]),
            "BB": np.random.randint(20, 60),
            "HR_last_10": np.random.randint(0, 5),
            "Cuota": round(np.random.uniform(5.0, 15.0), 2)
        })

# Convertimos a DataFrame
df = pd.DataFrame(rows)

# Calculamos HR_hoy con una fórmula simple
df["HR_hoy"] = df["HR_last_10"].apply(lambda x: np.random.binomial(1, min(0.3 * x / 10, 0.5)))

# Mostramos una muestra
print(df.head())
print("\nColumnas:", df.columns.tolist())
print("\nPromedio HR_hoy:", round(df["HR_hoy"].mean(), 3))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Variables de entrada(features)
X = df[["AVG_vs_P","HR_vs_P","BB","HR_last_10","Cuota"]]
# Variable objetivo
y = df["HR_hoy"]
# Separar en entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Entrenar el modelo
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(x_train,y_train)
# Evaluar el modelo
y_pred = model.predict(x_test)
print("Precision: ",round(accuracy_score(y_test,y_pred),3))
# Predecir probabilidad de HR
probas = model.predict_proba(x_test)[:,1] # Probabilidad de HR clase 1
# Calcular valor esperado = probabilidad X cuota
x_test = x_test.copy()
x_test["Prob_HR"] = probas
x_test["Cuota"] = df.loc[x_test.index, "Cuota"]
x_test["EV"] = x_test["Prob_HR"] * x_test["Cuota"]

print("\Top 5 apuestas con mayor valor esperado:")
print(x_test.sort_values("EV",ascending = False).head(5))



# Filtrar solo las apuestas con valor esperado mayor a 1
apuestas_valiosas = x_test[x_test["EV"] > 1]

# 5 Mejores apuestas individuales
mejores = apuestas_valiosas.sort_values("EV", ascending = False).head(5)
prob_combinada = mejores["Prob_HR"].prod()
cuota_combinada = mejores["Cuota"].prod()
EV_combinada = prob_combinada * cuota_combinada
print(mejores[["Cuota","Prob_HR", "EV"]])
print(f"Probabilidad total: {round(prob_combinada*100, 2)}%")
print(f"Cuota combinada: {round(cuota_combinada, 2)}")
print(f"EV combinada: {round(EV_combinada, 2)}")

X_test = x_test.copy()
X_test["Bateador"] = df.loc[X_test.index, "Bateador"]

mejores = apuestas_valiosas.sort_values("EV", ascending=False).head(5)
print(mejores[["Bateador", "Prob_HR", "Cuota", "EV"]])
prob_combinada = mejores["Prob_HR"].prod()
cuota_combinada = mejores["Cuota"].prod()
EV_combinada = prob_combinada * cuota_combinada

print(f"\n🧮 Probabilidad total: {round(prob_combinada * 100, 4)}%")
print(f"💰 Cuota combinada: {round(cuota_combinada, 2)}")
print(f"📈 EV combinado: {round(EV_combinada, 2)}")
