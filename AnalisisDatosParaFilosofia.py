import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Éste código calcula la regresión lineal para la matrícula total y de primer año de estudiantes de filosofía en la Universidad Alberto Hurtado (UAH) desde 2007 hasta 2024.
# El objetivo es predecir la matrícula total y de primer año para el año 2025. Asimismo, ver cómo ha variado la matrícula total y de primer año en el tiempo.
# Y también calcular la disminución de la matrícula total y de primer año, para ver si la matrícula de primer año disminuye más que la matrícula total,
# también para calcular la disminución de la matrícula total y de primer año en porcentaje y realizar una predicción 
# y saber si la carrera de filosofía en la UAH está en peligro o, incluso, si se podría cerrar la carrera en un futuro de 10 años y sostener que la filosofía está en crisis a largo plazo. 

#######
# IMPORTANTE: Todo éste cálculo está dedicado a la nota al pie número 29. 
# Los datos usados son reales y están disponibles en la página del Ministerio de Educación mediante Power BI (también están los datos de otras carreras, y 
# de la filosofía en total dentro de las universidades de Chile); fuente: https://www.mifuturo.cl/power-bi-matricula/ 
### Asimismo, el código realiza lo siguiente: 
### Se realiza una predicción para el año 2025 y se imprime el resultado en la terminal. Este resultado incluye graficar los datos y las líneas de regresión lineal.
# Ésto no tiene FrontEnd, es trabajo BackEnd (no hay interfaz gráfica como webs, sólo se imprime en la terminal), específicamente porque es para el análisis/ciencia de datos.
# Esto quiere decir que se debe ejecutar en la terminal de Python o en un IDE de Python (o un compilador de Python en general).
#######

# DETALLES TÉCNICOS Y METODOLÓGICOS (saltar comentario si no interesa):
# Se utiliza un modelo de regresión lineal simple para predecir la matrícula total y de primer año en el año 2025.
# Se calcula la pendiente (slope) e intercepto de la regresión lineal para la matrícula total y de primer año.
# Se usa un modelo de regresión lineal dado que los datos están registrados en una tendencia lineal, por lo que una regresión lineal múltiple no sería adecuada 
# (no hay dos o más variables independientes).
# Justificación de la regresión lineal: Sólo hay una única variable de predicción (año) y una única variable de respuesta (matrícula total o de primer año).
# Lo que, en ecuación, es así: (Y=mX+b); donde 'Y' es la matrícula total o de primer año, 'X' es el año, 'm' es la pendiente y 'b' es el intercepto (valor de Y cuando X=0).

#######
# 1. Regresión Lineal y Predicción (junto a declaración de variables)
#######
years = np.array([
    2007, 2008, 2009, 2010, 2011, 2012, 
    2013, 2014, 2015, 2016, 2017, 2018, 
    2019, 2020, 2021, 2022, 2023, 2024
])

matricula_total = np.array([
    116, 125, 126, 127,  107,  93, 
    107,  83,  76,  63,  61,  59, 
    60,  55,  61,  58,  52,  43  
])

matricula_1A = np.array([
    39,  46,  39,  42,  27,  24, 
    25,  24,  26,  25,  24,  20, 
    18,  14,  18,  11,   9,   6
])

X = years.reshape(-1, 1)

model_total = LinearRegression()
model_total.fit(X, matricula_total)

slope_total = model_total.coef_[0]
intercept_total = model_total.intercept_

model_1A = LinearRegression()
model_1A.fit(X, matricula_1A)

slope_1A = model_1A.coef_[0]
intercept_1A = model_1A.intercept_

# Resultados de la regresión
print("Regresión lineal - Matrícula Total")
print(f"  Pendiente (slope):  {slope_total:.2f}")
print(f"  Intercepto:         {intercept_total:.2f}")

print("\nRegresión lineal - Matrícula Primer Año")
print(f"  Pendiente (slope):  {slope_1A:.2f}")
print(f"  Intercepto:         {intercept_1A:.2f}")

# Predicción para 2025
prediccion_2025_total = model_total.predict([[2025]])
prediccion_2025_1A = model_1A.predict([[2025]])
print(f"\nPredicción matrícula total para 2025: {prediccion_2025_total[0]:.2f}") 
print(f"Predicción matrícula primer año para 2025: {prediccion_2025_1A[0]:.2f}")

#Cálculo de disminución porcentual entre 2007 y 2024

valor_inicial_total = matricula_total[0]

valor_final_total = matricula_total[-1]
disminucion_total = valor_inicial_total - valor_final_total
porcentaje_disminucion_total = (disminucion_total / valor_inicial_total) * 100

valor_inicial_1A = matricula_1A[0]
valor_final_1A = matricula_1A[-1]
disminucion_1A = valor_inicial_1A - valor_final_1A
porcentaje_disminucion_1A = (disminucion_1A / valor_inicial_1A) * 100

print(f"\nDisminución porcentual de matrícula total (2007-2024): {porcentaje_disminucion_total:.2f}%")
print(f"Disminución porcentual de matrícula primer año (2007-2024): {porcentaje_disminucion_1A:.2f}%")

########
# 2. Visualización de datos y líneas de regresión
########
# Se utiliza Matplotlib para generar un gráfico manipulable de los datos y las líneas de regresión lineal.
from sklearn.metrics import r2_score, mean_squared_error

# Evaluación del modelo para matrícula total (R² y MSE se incluyen antes del módulo 3 para evitar repetición de código)
pred_total_train = model_total.predict(X)
r2_total = r2_score(matricula_total, pred_total_train)
mse_total = mean_squared_error(matricula_total, pred_total_train)

# Evaluación del modelo para matrícula primer año
pred_1A_train = model_1A.predict(X)
r2_1A = r2_score(matricula_1A, pred_1A_train)
mse_1A = mean_squared_error(matricula_1A, pred_1A_train)

# Visualización de datos, líneas de regresión y métricas en el gráfico
plt.figure(figsize=(12, 6))

# 2.1 Datos para matrícula total y primer año (scatter)
plt.scatter(years, matricula_total, color='blue', label='Datos Matrícula Total')
plt.scatter(years, matricula_1A, color='red', label='Datos Matrícula 1A')

# 2.2 Línea de regresión para matrícula total
x_range = np.linspace(years[0], years[-1], 100)
y_total_line = slope_total * x_range + intercept_total
plt.plot(x_range, y_total_line, color='blue', linestyle='--', label='Regresión Total')

# 2.3 Línea de regresión para matrícula 1A
y_1A_line = slope_1A * x_range + intercept_1A
plt.plot(x_range, y_1A_line, color='red', linestyle='--', label='Regresión 1A')

# 2.4 Personalización del gráfico y contemplación de datos
plt.title('Visualización de Datos y Regresión Lineal')
plt.xlabel('Año')
plt.ylabel('Cantidad de Matrícula')
plt.legend()
plt.grid(True)

metrics_text = (
    f"Matrícula Total:\nR²: {r2_total:.4f}\nMSE: {mse_total:.4f}\n\n"
    f"Matrícula 1A:\nR²: {r2_1A:.4f}\nMSE: {mse_1A:.4f}"
)
plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()


########
# 3. Evaluación del modelo
########

# Evaluación del modelo para matrícula total
pred_total_train = model_total.predict(X)
r2_total = r2_score(matricula_total, pred_total_train)
mse_total = mean_squared_error(matricula_total, pred_total_train)

# Evaluación del modelo para matrícula primer año
pred_1A_train = model_1A.predict(X)
r2_1A = r2_score(matricula_1A, pred_1A_train)
mse_1A = mean_squared_error(matricula_1A, pred_1A_train)

# Impresión de métricas de calidad y ajuste
print("\nMétricas de calidad y ajuste:")
print("Matrícula Total:")
print(f"  Coeficiente de determinación (R²): {r2_total:.4f}")
print(f"  Error cuadrático medio (MSE): {mse_total:.4f}")

print("\nMatrícula Primer Año:")
print(f"  Coeficiente de determinación (R²): {r2_1A:.4f}")
print(f"  Error cuadrático medio (MSE): {mse_1A:.4f}")
