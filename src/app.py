from utils import db_connect
engine = db_connect()

# Your code here
import zipfile
import os

# Ruta del archivo ZIP
zip_path = "acea-water-prediction.zip"
extract_folder = "data"

# Extraer archivos
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print("Archivos extraídos en:", extract_folder)
import pandas as pd
import glob

# Ruta de los archivos CSV
csv_files = glob.glob(os.path.join(extract_folder, "*.csv"))

# Cargar todos los archivos en una lista de DataFrames
dataframes = [pd.read_csv(file) for file in csv_files]

# Unir todos los archivos en un solo DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Mostrar las primeras filas
print(df.head())
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
import matplotlib.pyplot as plt

# Seleccionar una variable para analizar
variable_a_graficar = "Rainfall_Gallicano"  # Ajusta según la columna que quieres visualizar

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df[variable_a_graficar], label=variable_a_graficar, color="b")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.title(f"Serie de tiempo: {variable_a_graficar}")
plt.legend()
plt.show()
df_model = df[["Date", "Rainfall_Gallicano"]].rename(columns={"Date": "ds", "Rainfall_Gallicano": "y"})
import pandas as pd

# Seleccionar la variable y la fecha
df_arima = df[["Date", "Rainfall_Gallicano"]].copy()
df_arima.set_index("Date", inplace=True)

# Verificar la estructura de los datos
print(df_arima.head())
df_arima.fillna(method="ffill", inplace=True)
print(df_arima.isnull().sum())  # Debe mostrar 0 si todos fueron corregidos
from statsmodels.tsa.stattools import adfuller

# Aplicar prueba de estacionaridad
adf_test = adfuller(df_arima["Rainfall_Gallicano"].dropna())

# Mostrar resultados
print(f"Estadístico ADF: {adf_test[0]}")
print(f"Valor p: {adf_test[1]}")
from statsmodels.tsa.arima.model import ARIMA

# Definir y entrenar el modelo ARIMA
model_arima = ARIMA(df_arima["Rainfall_Gallicano"], order=(2, 0, 2))  # Ajusta (p, d, q)
model_arima_fit = model_arima.fit()

# Generar pronóstico para los próximos 30 días
forecast_arima = model_arima_fit.forecast(steps=30)

# Mostrar predicción
print(forecast_arima)
import pickle

# Guardar el modelo ARIMA
with open("modelo_arima.pkl", "wb") as file:
    pickle.dump(model_arima_fit, file)

print("Modelo ARIMA guardado correctamente.")
