# 🏦 Predicción de Aprobación de Préstamos Bancarios (Loan Approval ML)

Aplicación web interactiva desarrollada en **Streamlit** como parte del Proyecto Final de **Machine Learning** (Maestría — Facultad de Ingeniería, Universidad Autónoma de Chihuahua).

Integra y compara en tiempo real los **tres modelos con mejor desempeño** evaluados en [`loan_approval.ipynb`](loan_approval.ipynb), utilizando el mismo preprocesamiento y orden de características aplicados durante el entrenamiento sobre [`loan_data.csv`](loan_data.csv).

---

## 🎓 Información Académica

- **Alumno:** Joel Arturo Álvarez Hernández (`403233`)
- **Institución:** Universidad Autónoma de Chihuahua (UACH) — Facultad de Ingeniería
- **Materia:** Machine Learning (Proyecto Final)
- **Asesora:** Dra. Olanda Prieto Ordaz

---

## 🧠 Modelos Implementados

Los tres modelos seleccionados del proyecto siguen un flujo de dos etapas en [`model_factory/train_model.py`](model_factory/train_model.py):
1. **Evaluación Baseline:** Entrenamiento y validación cruzada estratificada (5 folds) con parámetros por defecto.
2. **Ajuste de Hiperparámetros:** Búsqueda mediante `GridSearchCV` / `RandomizedSearchCV` (métrica objetivo `F1-Score`) y entrenamiento final sobre el 80% de entrenamiento (`36,000` registros), evaluando en el 20% de prueba (`9,000` registros).

Los tres pipelines completos (`ColumnTransformer` con `StandardScaler` + `OneHotEncoder` + estimador) se guardan por separado en `models/`:
- **SVM (Kernel RBF):** `models/svm_rbf.joblib`
- **Random Forest:** `models/random_forest.joblib`
- **Regresión Logística:** `models/logistic_regression.joblib`

---

## 🖥️ Funcionalidades de la Aplicación (`app.py`)

1. **Información del Proyecto:** Ficha académica, descripción del problema, base de datos (`loan_data.csv`) y modelos integrados.
2. **Predicción Individual:** Formulario validado con los 13 atributos del solicitante; muestra las predicciones de los 3 modelos lado a lado, probabilidades estimadas, score de decisión y análisis de consenso/diferencias.
3. **Upload Database:** Carga de archivos `.csv` con validación de columnas, tipos y categorías, ejecución por lotes de los 3 modelos y descarga de resultados.
4. **Evaluación Experimental:** Métricas reales (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`), curvas ROC y matrices de confusión en el conjunto de prueba (`models/evaluation_results.json`).
5. **Interpretación y Limitaciones:** Explicación clara de clases, probabilidades y limitaciones del conjunto de datos.

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
├── loan_data.csv                      # Base de datos (45,000 registros, 14 columnas)
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
