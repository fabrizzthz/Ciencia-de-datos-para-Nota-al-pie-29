import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tabulate import tabulate

# Éste código calcula la regresión lineal para la matrícula total y de primer año de estudiantes de filosofía en la Universidad Alberto Hurtado (UAH) desde 2007 hasta 2024.
# El objetivo es predecir la matrícula total y de primer año para el año 2025. Asimismo, ver cómo ha variado la matrícula total y de primer año en el tiempo.
# Y también calcular la disminución de la matrícula total y de primer año, y realizar una predicción 
# Y asú saber si la carrera de filosofía en la UAH está en peligro o, incluso, si se podría cerrar la carrera en un futuro de 10-15 años y sostener que la filosofía está en crisis a largo plazo. 

#######
# IMPORTANTE: Todo éste cálculo está dedicado a la nota al pie número 29. 
  # Los datos usados son reales y están disponibles en la página del Ministerio de Educación mediante (también están los datos de otras carreras, y 
  # de la filosofía en total dentro de las universidades de Chile); fuente: https://www.mifuturo.cl/power-bi-matricula/ 
### Asimismo, el código realiza lo siguiente: 
### Se realiza una predicción para el año 2025 y se imprime el resultado en la terminal. Este resultado incluye graficar los datos y las líneas de regresión lineal.
### También permite ver el gráfico según todos los años utilizados
# Ésto no tiene FrontEnd, es trabajo BackEnd (no hay interfaz gráfica como webs, sólo se imprime en la terminal), específicamente porque es para el análisis/ciencia de datos.
# Esto quiere decir que se debe ejecutar en la terminal de Python o en un IDE de Python (o un compilador de Python en general).
# Compilador en línea con soporte a librerías: https://www.mycompiler.io/es/new/python (sólo se debe copiar y pegar la totalidad del código)
#######

# DETALLES TÉCNICOS Y METODOLÓGICOS (saltar comentario si no interesa):
# Se utiliza un modelo de regresión lineal simple para predecir la matrícula total y de primer año en el año 2025.
# Se usa un modelo de regresión lineal dado que los datos están registrados en una tendencia lineal, por lo que una regresión lineal múltiple no sería adecuada 
# (no hay dos o más variables independientes).
# Justificación de la regresión lineal: Sólo hay una única variable de predicción (año) y una única variable de respuesta (matrícula total o de primer año).

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "DejaVu Sans"

YEARS = np.array([2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
                2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

MATRICULA_TOTAL = np.array([116, 125, 126, 127, 107, 93, 107, 83,
                           76, 63, 61, 59, 60, 55, 61, 58, 52, 43])

MATRICULA_1A = np.array([39, 46, 39, 42, 27, 24, 25, 24,
                        26, 25, 24, 20, 18, 14, 18, 11, 9, 6])

CONFIG = {
    "polynomial_degree": 2,
    "test_size": 3,  
    "alphas": [0.1, 1, 10], 
    "prediction_year": 2025
}

class EnrollmentPredictor:
    """Clase para modelado y predicción de matrículas"""
    
    def __init__(self, years, enrollments, config):
        self.years = years
        self.enrollments = enrollments
        self.config = config
        self.model = None
        self.poly = PolynomialFeatures(config["polynomial_degree"], include_bias=False)
        self.best_alpha = None
        
    def _prepare_data(self):
        """Prepara datos para entrenamiento con características polinómicas y lag"""
        X_lag = self.enrollments[:-1]
        y_lag = self.years[1:]
        target = self.enrollments[1:]
        
        X_poly = self.poly.fit_transform(y_lag.reshape(-1, 1))
        return np.hstack([X_poly, X_lag.reshape(-1, 1)]), target
    
    def _tune_hyperparameters(self, X, y):
        """Optimiza hiperparámetros usando validación cruzada temporal"""
        tscv = TimeSeriesSplit(n_splits=3)
        model = Ridge()
        grid = GridSearchCV(model, {"alpha": self.config["alphas"]}, 
                          cv=tscv, scoring="neg_mean_squared_error")
        grid.fit(X, y)
        self.best_alpha = grid.best_params_["alpha"]
        return grid.best_estimator_
    
    def train(self):
        """Entrena el modelo con optimización de hiperparámetros"""
        X, y = self._prepare_data()
        self.model = self._tune_hyperparameters(X, y)
        self.model.fit(X, y)
        return self
    
    def predict(self, year, last_enrollment):
        """Realiza predicción para un año específico"""
        X_poly = self.poly.transform([[year]])
        X_combined = np.hstack([X_poly, [[last_enrollment]]])
        return self.model.predict(X_combined)[0]
    
    def evaluate(self):
        """Evalúa el modelo y devuelve métricas"""
        X, y = self._prepare_data()
        preds = self.model.predict(X)
        return {
            "mse": mean_squared_error(y, preds),
            "r2": r2_score(y, preds),
            "coefs": self.model.coef_
        }

def generate_predictions_plot(predictor_total, predictor_1A, pred_2025_total, pred_2025_1A):
    """Genera visualización profesional de resultados"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("Año", fontsize=12)
        ax.set_ylabel("Matrícula", fontsize=12)
    
# Gráfico para Matrícula Total
    ax1.scatter(YEARS, MATRICULA_TOTAL, color="navy", label="Datos Históricos")
    ax1.plot(YEARS[1:], predictor_total.model.predict(predictor_total._prepare_data()[0]), 
            color="dodgerblue", linestyle="--", label="Tendencia Modelada")
    ax1.scatter(CONFIG["prediction_year"], pred_2025_total, 
               color="red", marker="X", s=100, label="Predicción 2025")
    ax1.set_title("Matrícula Total - Modelado y Predicción", fontsize=14, pad=20)
    ax1.legend()
    
#Gráfico para Matrícula primer año (o 1A)
    ax2.scatter(YEARS, MATRICULA_1A, color="darkgreen", label="Datos Históricos")
    ax2.plot(YEARS[1:], predictor_1A.model.predict(predictor_1A._prepare_data()[0]), 
            color="limegreen", linestyle="--", label="Tendencia Modelada")
    ax2.scatter(CONFIG["prediction_year"], pred_2025_1A, 
              color="red", marker="X", s=100, label="Predicción 2025")
    ax2.set_title("Matrícula de Primer Año - Modelado y Predicción", fontsize=14, pad=20)
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    predictor_total = EnrollmentPredictor(YEARS, MATRICULA_TOTAL, CONFIG).train()
    predictor_1A = EnrollmentPredictor(YEARS, MATRICULA_1A, CONFIG).train()
    pred_2025_total = predictor_total.predict(CONFIG["prediction_year"], MATRICULA_TOTAL[-1])
    pred_2025_1A = predictor_1A.predict(CONFIG["prediction_year"], MATRICULA_1A[-1])
    metrics_total = predictor_total.evaluate()
    metrics_1A = predictor_1A.evaluate()
    
    table = [
        ["Matrícula Total", metrics_total["mse"], metrics_total["r2"], pred_2025_total],
        ["Matrícula Primer Año", metrics_1A["mse"], metrics_1A["r2"], pred_2025_1A]
    ]
    print(tabulate(table, headers=["Métrica", "MSE", "R²", "Predicción 2025"], 
                 tablefmt="grid", floatfmt=(".0f", ".2f", ".2f", ".1f")))
    
    fig = generate_predictions_plot(predictor_total, predictor_1A, pred_2025_total, pred_2025_1A)
    plt.show()
