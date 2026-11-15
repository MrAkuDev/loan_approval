# 🏦 Clasificación de Riesgo Crediticio y Aprobación de Préstamos (Credit Risk ML)

Aplicación web interactiva desarrollada en **Streamlit** como parte del Proyecto Final de **Machine Learning** (Maestría — Facultad de Ingeniería, Universidad Autónoma de Chihuahua).

Integra y compara en tiempo real **tres modelos supervisados** (`SVM RBF`, `Random Forest` y `Regresión Logística`), utilizando un preprocesamiento unificado con imputación y escalamiento/codificación sin fuga de datos sobre [`credit_risk_dataset.csv`](credit_risk_dataset.csv) (*Kaggle — laotse/credit-risk-dataset*).

---

## 🎓 Información Académica

- **Alumno:** Joel Arturo Álvarez Hernández (`403233`)
- **Institución:** Universidad Autónoma de Chihuahua (UACH) — Facultad de Ingeniería
- **Materia:** Machine Learning (Proyecto Final)
- **Asesora:** Dra. Olanda Prieto Ordaz

---

## 🧠 Modelos Implementados

Los tres modelos seleccionados del proyecto siguen un flujo modular en [`model_factory/train_model.py`](model_factory/train_model.py):
1. **Limpieza y Partición Estratificada:** Deduplicación (`165` duplicados) y filtrado determinista de atípicos imposibles (`person_age > 100`, `person_emp_length > 60`) sobre `credit_risk_dataset.csv`, manteniendo `32,409` registros limpios divididos 80% entrenamiento (`25,927`) / 20% prueba (`6,482`).
2. **Evaluación Baseline:** Validación cruzada estratificada (5 folds) con parámetros por defecto.
3. **Ajuste de Hiperparámetros:** Búsqueda mediante `GridSearchCV` / `RandomizedSearchCV` (métrica objetivo `F1-Score`) y entrenamiento final sobre el 100% de `X_train`.

Los tres pipelines completos (`ColumnTransformer` con `SimpleImputer` + `StandardScaler` + `OneHotEncoder` + estimador) se guardan por separado en `models/`:
- **SVM (Kernel RBF):** `models/svm_rbf.joblib`
- **Random Forest:** `models/random_forest.joblib`
- **Regresión Logística:** `models/logistic_regression.joblib`

---

## 🖥️ Funcionalidades de la Aplicación (`app.py`)

1. **Inicio (Información y Predicción Individual):** Ficha académica, detalles del dataset (`credit_risk_dataset.csv`) y formulario validado con los 11 atributos del solicitante; muestra las predicciones (`Sin Incumplimiento (0)` vs `Incumplimiento (1)`) de los 3 modelos lado a lado, probabilidades estimadas, explicación local Shapley y análisis de consenso/diferencias.
2. **Lab — Local (Evaluación Experimental y EDA):** Correlaciones sin fuga de datos, EDA en `StratifiedKFold (cv_5)`, vista 3D interactiva, fronteras de decisión 2D, heatmaps de hiperparámetros, curvas ROC y matrices de confusión en Test (`models/evaluation_results.json`).
3. **Lab — Try yours (Upload Database CSV):** Carga de archivos `.csv` con validación de columnas, tipos y categorías (incluyendo imputación automática de nulos en `Pipeline`), ejecución por lotes de los 3 modelos y descarga de resultados.
4. **Acerca de (Interpretación y Limitaciones):** Guía clara de clases (`Sin Incumplimiento (0)`, `Incumplimiento (1)`), probabilidades y limitaciones del conjunto de datos.

---

## 📁 Estructura del Repositorio

```text
├── app.py                             # Punto de entrada ligero que delega en app/main.py
├── app/                               # Paquete modular de la interfaz Streamlit
│   ├── __init__.py
│   ├── config.py                      # Rutas, esquema de columnas y paleta de colores
│   ├── validation.py                  # Validación de instancias individuales y CSV
│   ├── services.py                    # Carga cacheada de modelos/datos e inferencia
│   ├── frontend.py                    # Configuración de página, CSS, sidebar y tarjetas
│   ├── charts.py                      # Gráficas 2D (Matplotlib) de predicción, EDA y Test
│   ├── charts_3d.py                   # Gráfica 3D interactiva HTML5 Canvas
│   ├── tab_info.py                    # Pestaña 1: Información del proyecto y modelos
│   ├── tab_predict.py                 # Pestaña 2: Predicción individual
│   ├── tab_upload.py                  # Pestaña 3: Predicción por lotes (CSV)
│   ├── tab_eval.py                    # Pestaña 4: Evaluación experimental y EDA (cv_5)
│   ├── tab_interp.py                  # Pestaña 5: Interpretación y limitaciones
│   └── main.py                        # Orquestador principal renderizado por app.py
├── model_factory/                     # Paquete modular de entrenamiento y evaluación
│   ├── __init__.py
│   ├── config.py                      # Rutas, esquema de variables y espacios de búsqueda
│   ├── data.py                        # Carga de CSV, partición Train/Test y correlaciones
│   ├── preprocessing.py               # ColumnTransformer y ensamblado de Pipeline
│   ├── evaluation.py                  # Métricas CV baseline y evaluación en Test
│   ├── tuning.py                      # Búsqueda de hiperparámetros y exportación .joblib
│   └── train_model.py                 # Punto de entrada principal del entrenamiento
├── requirements.txt                   # Dependencias para despliegue en Streamlit Community Cloud
├── loan_approval.ipynb                # Notebook experimental del proyecto final
├── credit_risk_dataset.csv            # Base de datos Kaggle (32,581 filas brutas, 12 columnas)
├── models/
│   ├── svm_rbf.joblib                 # Pipeline entrenado SVM RBF
│   ├── random_forest.joblib           # Pipeline entrenado Random Forest
│   ├── logistic_regression.joblib     # Pipeline entrenado Regresión Logística
│   └── evaluation_results.json        # Métricas experimentales reales (CV y Test Set)
├── gg.md                              # Especificación de requisitos del proyecto
└── README.md                          # Documentación del proyecto
```

---

## 💻 Ejecución Local

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. (Opcional) Re-entrenar y exportar los 3 modelos
```bash
python model_factory/train_model.py
```

### 3. Ejecutar la aplicación Streamlit
```bash
python -m streamlit run app.py
```

### 4. Despliegue en Streamlit Community Cloud
Configurar la app vinculada a este repositorio en GitHub (`MrAkuDev/loan_approval`), rama `main`, archivo principal `app.py`.
