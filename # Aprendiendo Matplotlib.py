# Aprendiendo Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
# Grafico de barras (barplot)
data = {
    "Jugador": ["Soto","Judge","Betts","Ohtani","Alonso"],
    "HR": [18,27,21,23,25]
}
df = pd.DataFrame(data)
sns.barplot(x="Jugador", y="HR", data=df)
plt.title("Home runs por jugador")
plt.show()

# Grafico de dispersion (scatterplot)
# Ideal para ver si mas BB se relacionan con mas HR
df["BB"] = [40,50,38,45,33]
sns.scatterplot(x="BB", y="HR", data=df)
plt.title("HR vs BB")
plt.xlabel("Boletos")
plt.ylabel("Home runs")
plt.show()

# Histograma para distribucion de AVG
df["AVG"] = [0.285, 0.300, 0.278, 0.290, 0.260]
plt.hist(df["AVG"],bins = 5,color="orange")
plt.title("Distribucion de AVG")
plt.xlabel("AVG")
plt.ylabel("Frecuencia")
plt.show()

# Heatmap de correlacion
corr = df[["HR","BB","AVG"]].corr()
sns.heatmap(corr,annot = True,cmap="coolwarm")
plt.title("Correlacion entre variables")
plt.show()

# Ejercicio 5
